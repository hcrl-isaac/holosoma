"""Robust stance windows for g1fk sources: position+height rule + a locomotion support invariant.

The stock detector (source toe xy-speed < 1 cm/s) finds nothing on IK-retargeted sources whose stance
feet skate at 1-5 cm/s. This computes, on the 30 fps source the solver sees:

1. Per-toe stance candidates: in-window travel < ``disp_tol`` AND toe within ``height_tol`` of the
   terrain surface under it (court cuboids + ground) -- Rempe-style position+height.
2. Support invariant: any frame where NEITHER foot qualifies and the clip is not truly airborne
   (best toe clearance < ``flight_tol``) gets its better-grounded foot forced into stance -- for
   locomotion at least one foot is in contact at all times, even when the dirty source hovers.
3. Contiguous stance runs become z-lock windows ``(start, end, z_anchor)`` where the anchor is the
   window's surface height plus the clip's own calibrated toe-center offset -- the constraint that
   actually pulls a hovering source foot DOWN onto the terrain (xy sticking alone cannot).

Saved as ``<seq_dir>/<stem>_foot_sticking.npz``: ``sticking`` (T, 2) bool for xy-stick, plus
``windows_left`` / ``windows_right`` float arrays (n, 5): (start, end, x, y, z) source anchors for the retargeter's per-window foot z-lock.
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


def _still_mask(toe: np.ndarray, disp_tol: float, half_win: int) -> np.ndarray:
    t_n = len(toe)
    still = np.zeros(t_n, dtype=bool)
    for t in range(t_n):
        lo, hi = max(0, t - half_win), min(t_n, t + half_win + 1)
        seg = toe[lo:hi]
        still[t] = float(np.linalg.norm(seg - seg.mean(0), axis=1).max()) < disp_tol
    return still


def _smooth(mask: np.ndarray, min_run: int = 3, max_gap: int = 2) -> np.ndarray:
    """Close short gaps, then drop runs shorter than min_run."""
    m = mask.copy()
    t_n = len(m)
    # close gaps
    t = 0
    while t < t_n:
        if not m[t] and t > 0 and m[t - 1]:
            g = t
            while g < t_n and not m[g]:
                g += 1
            if g < t_n and (g - t) <= max_gap:
                m[t:g] = True
            t = g
        else:
            t += 1
    # drop short runs
    t = 0
    while t < t_n:
        if m[t]:
            e = t
            while e + 1 < t_n and m[e + 1]:
                e += 1
            if (e - t + 1) < min_run:
                m[t : e + 1] = False
            t = e + 1
        else:
            t += 1
    return m


def compute(src: np.ndarray, boxes: np.ndarray, disp_tol: float, height_tol: float, flight_tol: float,
            half_win: int) -> tuple[np.ndarray, list, list]:
    """Stance masks (T, 2) + per-foot z-lock windows [(start, end, z_anchor)] for one source clip."""
    toe_idx = [G1FK_DEMO_JOINTS.index(n) for n in TOE_NAMES_BY_FORMAT["g1fk"]]
    toes = src[:, toe_idx]  # (T, 2, 3)
    surf = np.stack([terrain_z(toes[:, k, :2], boxes) for k in range(2)], axis=1)  # (T, 2)
    clear = toes[:, :, 2] - surf
    speed = np.zeros_like(clear)
    speed[1:] = np.linalg.norm(np.diff(toes[:, :, :2], axis=0), axis=2)

    cand = np.stack(
        [_still_mask(toes[:, k], disp_tol, half_win) & (clear[:, k] < height_tol) for k in range(2)], axis=1
    )
    # support invariant: unless truly airborne, the better-grounded foot is in stance
    for t in range(len(cand)):
        if not cand[t].any() and clear[t].min() < flight_tol:
            k = int(np.argmin(clear[t] + speed[t]))
            cand[t, k] = True
    masks = np.stack([_smooth(cand[:, k]) for k in range(2)], axis=1)

    # calibrated toe-center offset above the surface when planted (per clip; sphere radius + skin)
    planted = clear[masks]
    offset = float(np.clip(np.percentile(planted, 20), 0.008, 0.05)) if planted.size else 0.02  # floor: sphere r=5mm + margin

    windows: list[list] = [[], []]
    for k in range(2):
        t = 0
        m = masks[:, k]
        while t < len(m):
            if m[t]:
                e = t
                while e + 1 < len(m) and m[e + 1]:
                    e += 1
                # full 3D anchor from the SOURCE stance: xy = where the source foot actually plants
                # (anchoring to the OUTPUT's own position freezes a lagging foot mid-flight), z = the
                # surface under that spot + calibrated toe offset
                x_a = float(np.median(toes[t : e + 1, k, 0]))
                y_a = float(np.median(toes[t : e + 1, k, 1]))
                z_a = float(terrain_z(np.array([[x_a, y_a]]), boxes)[0]) + offset
                windows[k].append([t, e, x_a, y_a, z_a])
                t = e + 1
            else:
                t += 1
    return masks, windows[0], windows[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq_dirs", nargs="+", required=True, help="Seq dir(s) holding <stem>.npy sources.")
    ap.add_argument("--courts_json", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--disp_tol", type=float, default=0.04, help="Max in-window travel (m) for stance.")
    ap.add_argument("--height_tol", type=float, default=0.09, help="Max toe clearance (m) for stance.")
    ap.add_argument("--flight_tol", type=float, default=0.15, help="Min best-toe clearance (m) for true flight.")
    ap.add_argument("--half_win", type=int, default=2, help="Half window (solver frames, 30 fps).")
    args = ap.parse_args()

    court = json.loads(Path(args.courts_json).read_text())["courts"][args.court]
    boxes = np.array([[*p["pos"], *p["size"]] for p in court["prims"]], dtype=np.float64)
    for d in args.seq_dirs:
        seq = Path(d)
        src = np.load(seq / f"{seq.name}.npy")[::DOWNSAMPLE]
        masks, wl, wr = compute(src, boxes, args.disp_tol, args.height_tol, args.flight_tol, args.half_win)
        np.savez(
            seq / f"{seq.name}_foot_sticking.npz",
            sticking=masks,
            toe_names=TOE_NAMES_BY_FORMAT["g1fk"],
            windows_left=np.array(wl, dtype=float).reshape(-1, 5),
            windows_right=np.array(wr, dtype=float).reshape(-1, 5),
        )
        both_off = (~masks.any(axis=1)).mean()
        print(
            f"[stance] {seq.name}: L {masks[:, 0].mean() * 100:.0f}% / R {masks[:, 1].mean() * 100:.0f}% | "
            f"no-support frames {both_off * 100:.0f}% | windows L{len(wl)}/R{len(wr)}"
        )


if __name__ == "__main__":
    main()
