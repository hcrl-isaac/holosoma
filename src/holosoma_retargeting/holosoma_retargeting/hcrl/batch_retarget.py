"""Corpus batch driver: manifest + fitted courts + normalized csvs -> retargeted csvs + per-clip report.

Three stages per clip (grouped by court):
  1. prep (serial, cheap): FK the source csv into the seq dir (``csv_to_g1fk``), generate the court
     terrain files (``courts_to_scene``), compute 3D-anchored stance windows (``stance_windows``).
  2. solve (parallel): one ``examples/robot_retarget.py`` subprocess per clip (the sequential SQP is
     single-threaded; clips are the natural parallel unit).
  3. export + score (serial, fast): ``qpos_to_csv`` back to the BONES layout at 120 fps, then a
     support-coverage metric (fraction of frames with a foot sphere within 2 cm of the court surface,
     longest airborne stretch) via mujoco FK of the output.

Writes ``<report>`` csv with per-clip: court, solve rc, final cost, support coverage, max airborne
stretch, output path -- the quality gate for the bundle build filters on these columns.
"""

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from holosoma_retargeting.config_types.data_type import G1FK_DEMO_JOINTS, TOE_NAMES_BY_FORMAT
from holosoma_retargeting.hcrl import courts_to_scene, stance_windows
from holosoma_retargeting.hcrl.csv_to_g1fk import DEFAULT_MODEL, fk_positions
from holosoma_retargeting.hcrl.qpos_to_csv import qpos_to_csv

PKG = Path(__file__).resolve().parents[1]
COURT_REMAP = {"stairs_r013": "stairs_env"}  # stairs stage rebuilt the court under a new name


def prep_clip(csv_path: Path, court: dict, court_boxes: np.ndarray, seq_dir: Path, model: mujoco.MjModel,
              robot_xml: Path) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / f"{seq_dir.name}.npy", fk_positions(model, csv_path))
    courts_to_scene.build_court_files(court, seq_dir, robot_xml)
    src = np.load(seq_dir / f"{seq_dir.name}.npy")[:: stance_windows.DOWNSAMPLE]
    masks, wl, wr = stance_windows.compute(src, court_boxes, 0.04, 0.09, 0.15, 2)
    np.savez(
        seq_dir / f"{seq_dir.name}_foot_sticking.npz",
        sticking=masks,
        toe_names=TOE_NAMES_BY_FORMAT["g1fk"],
        windows_left=np.array(wl, dtype=float).reshape(-1, 5),
        windows_right=np.array(wr, dtype=float).reshape(-1, 5),
    )


def solve_clip(args: tuple) -> tuple:
    """Subprocess one retarget; returns (stem, rc, cost)."""
    stem, data_root, save_dir = args
    cmd = [
        sys.executable, str(PKG / "examples" / "robot_retarget.py"),
        "--data_path", str(data_root), "--task-type", "climbing", "--task-name", stem,
        "--data_format", "g1fk",
        "--robot-config.robot-urdf-file", str(PKG / "models" / "g1" / "g1_29dof_spherehand.urdf"),
        "--task-config.object-dir", str(Path(data_root) / stem),
        "--save_dir", str(save_dir),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(PKG))
        cost = None
        for tok in reversed(out.stdout.replace("\r", "\n").split()):
            if tok.startswith("cost="):
                cost = float(tok.split("=")[1].rstrip("]"))
                break
        return stem, out.returncode, cost
    except subprocess.TimeoutExpired:
        return stem, -9, None


def support_coverage(qpos: np.ndarray, boxes: np.ndarray, model: mujoco.MjModel) -> tuple[float, float]:
    """(fraction of frames with a foot sphere within 2 cm of the surface, longest airborne stretch s)."""
    data = mujoco.MjData(model)
    sph = [[model.body(f"{s}_ankle_roll_sphere_{i}_link").id for i in range(1, 6)] for s in ("left", "right")]
    minc = np.zeros(len(qpos))
    for t, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        cl = [min(data.xpos[i][2] - stance_windows.terrain_z(data.xpos[i][None, :2], boxes)[0] for i in side) for side in sph]
        minc[t] = min(cl) - 0.005
    airborne = minc >= 0.02
    longest = 0
    run = 0
    for a in airborne:
        run = run + 1 if a else 0
        longest = max(longest, run)
    return float(1.0 - airborne.mean()), longest / 30.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", required=True, help="Normalized clip csvs (csv_norm).")
    ap.add_argument("--manifest", required=True, help="Binned manifest (clip -> court).")
    ap.add_argument("--courts_json", required=True)
    ap.add_argument("--work_root", required=True, help="Root for seq dirs + solver outputs.")
    ap.add_argument("--export_dir", required=True, help="Final retargeted csvs (BONES layout, 120 fps).")
    ap.add_argument("--report", required=True, help="Per-clip results csv.")
    ap.add_argument("--courts", nargs="*", default=None, help="Only these courts (default: all box/edge/stairs).")
    ap.add_argument("--limit", type=int, default=None, help="Cap clips per court (smoke runs).")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    courts = json.loads(Path(args.courts_json).read_text())["courts"]
    man = pd.read_csv(args.manifest)
    man["stem"] = man["move_g1_path"].map(lambda p: Path(p).stem)
    man["court"] = man["court"].map(lambda c: COURT_REMAP.get(c, c))
    rows = man[man["court"].isin(set(courts) if args.courts is None else set(args.courts))]
    csv_dir = Path(args.csv_dir)
    work = Path(args.work_root)
    data_root, save_dir = work / "seq", work / "qpos"
    save_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir)

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    robot_xml = PKG / "models" / "g1" / "g1_29dof_spherehand.xml"

    # stage 1: prep
    todo = []
    for court_name, grp in rows.groupby("court"):
        stems = [s for s in grp["stem"] if (csv_dir / f"{s}.csv").exists()][: args.limit]
        boxes = np.array([[*p["pos"], *p["size"]] for p in courts[court_name]["prims"]], dtype=np.float64)
        for stem in stems:
            prep_clip(csv_dir / f"{stem}.csv", courts[court_name], boxes, data_root / stem, model, robot_xml)
            todo.append((stem, court_name))
        print(f"[batch] prepped {len(stems)} clips on {court_name}")

    # stage 2: parallel solves
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(solve_clip, (stem, data_root, save_dir)): stem for stem, _ in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            stem, rc, cost = fut.result()
            results[stem] = (rc, cost)
            print(f"[batch] {i}/{len(todo)} {stem}: rc={rc} cost={cost}")

    # stage 3: export + score
    export_dir.mkdir(parents=True, exist_ok=True)
    columns = None
    report_rows = []
    for stem, court_name in todo:
        rc, cost = results[stem]
        rec = {"stem": stem, "court": court_name, "rc": rc, "cost": cost,
               "support": None, "max_air_s": None, "out": ""}
        npz = save_dir / f"{stem}_original.npz"
        if rc == 0 and npz.exists():
            q = np.load(npz, allow_pickle=True)["qpos"]
            boxes = np.array([[*p["pos"], *p["size"]] for p in courts[court_name]["prims"]], dtype=np.float64)
            rec["support"], rec["max_air_s"] = support_coverage(q, boxes, model)
            if columns is None:
                columns = list(pd.read_csv(csv_dir / f"{stem}.csv", nrows=0).columns)
            df = qpos_to_csv(np.asarray(q), model, columns, 4)
            out_path = export_dir / f"{stem}.csv"
            df.to_csv(out_path, index=False)
            rec["out"] = str(out_path)
        report_rows.append(rec)

    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)
    ok = [r for r in report_rows if r["rc"] == 0]
    sup = [r["support"] for r in ok if r["support"] is not None]
    print(f"[batch] done: {len(ok)}/{len(report_rows)} solved | support med "
          f"{np.median(sup) * 100:.0f}% min {min(sup) * 100:.0f}%" if sup else "[batch] no successes")


if __name__ == "__main__":
    main()
