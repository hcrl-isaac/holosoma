"""Convert BONES-SEED-layout G1 csvs into g1fk climbing-task source data (one seq dir per clip).

Each csv row is (px py pz qx qy qz qw + 29 joint angles); mujoco FK on the holosoma G1 model turns it
into world positions of the ``G1FK_DEMO_JOINTS`` links, saved as ``<out_root>/<stem>/<stem>.npy`` with
shape (T, 15, 3) -- the exact source format the climbing task loads (it downsamples x4 internally).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from holosoma_retargeting.config_types.data_type import G1FK_DEMO_JOINTS

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PKG_ROOT / "models" / "g1" / "g1_29dof_spherehand.xml"


def qpos_row(model: mujoco.MjModel, csv_path: Path, row_idx: int = 0) -> np.ndarray:
    """One csv row as a mujoco qpos vector (freejoint xyz + wxyz, then joints in model order)."""
    df = pd.read_csv(csv_path, nrows=row_idx + 1)
    row = df.to_numpy(dtype=np.float64)[row_idx]
    joint_cols = list(df.columns[7:])
    q = np.zeros(model.nq)
    q[:3] = row[:3]
    q[3:7] = (row[6], row[3], row[4], row[5])  # xyzw -> wxyz
    for name, val in zip(joint_cols, row[7:]):
        for j in range(model.njnt):
            if model.joint(j).name == name:
                q[model.jnt_qposadr[j]] = val
                break
    return q


def fk_positions(model: mujoco.MjModel, csv_path: Path) -> np.ndarray:
    """FK a csv clip into (T, len(G1FK_DEMO_JOINTS), 3) world link positions."""
    df = pd.read_csv(csv_path)
    joint_cols = list(df.columns[7:])
    # qpos layout: freejoint (xyz + wxyz) then hinge joints in model order
    hinge_addr = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in joint_cols:
            hinge_addr[name] = model.jnt_qposadr[j]
    missing = [c for c in joint_cols if c not in hinge_addr]
    if missing:
        raise ValueError(f"csv joints missing from model: {missing}")
    body_ids = [model.body(n).id for n in G1FK_DEMO_JOINTS]

    data = mujoco.MjData(model)
    vals = df.to_numpy(dtype=np.float64)
    out = np.zeros((len(vals), len(body_ids), 3), dtype=np.float32)
    for t, row in enumerate(vals):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = (row[6], row[3], row[4], row[5])  # xyzw -> wxyz
        for name, col in zip(joint_cols, row[7:]):
            data.qpos[hinge_addr[name]] = col
        mujoco.mj_forward(model, data)
        out[t] = data.xpos[body_ids]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True, help="Input clip csv(s) (BONES-SEED layout).")
    ap.add_argument("--out_root", required=True, help="Climbing data root; one <stem>/ seq dir per clip.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="G1 mujoco xml used for FK.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    out_root = Path(args.out_root)
    for c in args.csv:
        csv_path = Path(c)
        seq_dir = out_root / csv_path.stem
        seq_dir.mkdir(parents=True, exist_ok=True)
        pos = fk_positions(model, csv_path)
        np.save(seq_dir / f"{csv_path.stem}.npy", pos)
        np.save(seq_dir / f"{csv_path.stem}_q0.npy", qpos_row(model, csv_path))
        print(f"[g1fk] {csv_path.stem}: {pos.shape} -> {seq_dir}")


if __name__ == "__main__":
    main()
