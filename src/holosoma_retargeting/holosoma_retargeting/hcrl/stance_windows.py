"""Robust stance windows for g1fk sources: position+height rule against the court geometry.

The stock detector (source toe xy-speed < 1 cm/s) finds nothing on IK-retargeted sources whose stance
feet skate at 1-5 cm/s. Rempe et al. (ECCV 2020) rule instead: a toe is in stance when it (a) moves
less than ``disp_tol`` within a centered window AND (b) sits within ``height_tol`` of the terrain
surface under it (court cuboids + ground). Windows are computed on the 30 fps source the solver sees
and saved as ``<seq_dir>/<stem>_foot_sticking.npz`` (bool (T, 2) for [left, right] toes), which
``robot_retarget.py`` picks up in place of the velocity heuristic.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.data_type import G1FK_DEMO_JOINTS, TOE_NAMES_BY_FORMAT

DOWNSAMPLE = 4  # climbing task downsamples the 120 fps source x4; windows must match what it solves on


def terrain_z(xy: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Surface height under each xy: highest covering cuboid top, else ground 0. xy: (F, 2)."""
    if not len(boxes):
        return np.zeros(len(xy))
    cx, cy, cz, sx, sy, sz = boxes.T
    inside = (np.abs(xy[:, None, 0] - cx[None]) <= sx[None] / 2) & (np.abs(xy[:, None, 1] - cy[None]) <= sy[None] / 2)
    covered = np.where(inside, (cz + sz / 2)[None], -np.inf)
    z = covered.max(axis=1)
    return np.where(np.isfinite(z), z, 0.0)


def stance_mask(toe: np.ndarray, boxes: np.ndarray, disp_tol: float, height_tol: float, half_win: int) -> np.ndarray:
    """Per-frame stance bool for one toe trajectory (T, 3) at solver rate."""
    t_n = len(toe)
    still = np.zeros(t_n, dtype=bool)
    for t in range(t_n):
        lo, hi = max(0, t - half_win), min(t_n, t + half_win + 1)
        seg = toe[lo:hi]
        still[t] = float(np.linalg.norm(seg - seg.mean(0), axis=1).max()) < disp_tol
    near = (toe[:, 2] - terrain_z(toe[:, :2], boxes)) < height_tol
    mask = still & near
    # close 1-frame gaps, then drop 1-frame blips
    for t in range(1, t_n - 1):
        if mask[t - 1] and mask[t + 1]:
            mask[t] = True
    for t in range(t_n):
        if mask[t] and not (t > 0 and mask[t - 1]) and not (t + 1 < t_n and mask[t + 1]):
            mask[t] = False
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq_dirs", nargs="+", required=True, help="Seq dir(s) holding <stem>.npy sources.")
    ap.add_argument("--courts_json", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--disp_tol", type=float, default=0.025, help="Max in-window travel (m) for stance.")
    ap.add_argument("--height_tol", type=float, default=0.06, help="Max toe height above surface (m).")
    ap.add_argument("--half_win", type=int, default=2, help="Half window (solver frames, 30 fps).")
    args = ap.parse_args()

    court = json.loads(Path(args.courts_json).read_text())["courts"][args.court]
    boxes = np.array([[*p["pos"], *p["size"]] for p in court["prims"]], dtype=np.float64)
    toe_idx = [G1FK_DEMO_JOINTS.index(n) for n in TOE_NAMES_BY_FORMAT["g1fk"]]
    for d in args.seq_dirs:
        seq = Path(d)
        src = np.load(seq / f"{seq.name}.npy")[::DOWNSAMPLE]
        masks = np.stack([stance_mask(src[:, k], boxes, args.disp_tol, args.height_tol, args.half_win) for k in toe_idx], axis=1)
        np.savez(seq / f"{seq.name}_foot_sticking.npz", sticking=masks, toe_names=TOE_NAMES_BY_FORMAT["g1fk"])
        frac = masks.mean(0)
        print(f"[stance] {seq.name}: L {frac[0] * 100:.0f}% / R {frac[1] * 100:.0f}% of {len(masks)} frames in stance")


if __name__ == "__main__":
    main()
