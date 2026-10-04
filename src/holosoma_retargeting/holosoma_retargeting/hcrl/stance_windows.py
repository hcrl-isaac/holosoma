"""Stance windows for g1fk sources: a support-state labeling against the terrain under each toe.

The stock detector (source toe xy-speed < 1 cm/s) finds nothing on IK-retargeted sources whose stance
feet skate at 1-5 cm/s. On the 30 fps source the solver sees, this labels every frame both / left /
right / flight with a Viterbi pass whose emissions are toe heights above the terrain (court cuboids +
ground) and whose switching penalty keeps a noisy foot in its current state.

Each stance run becomes a 3D anchor window ``(start, end, x, y, z)``: xy where the source foot plants,
z the surface there plus the clip's calibrated toe offset, which pulls a hovering foot down onto it.
Saved as ``<seq_dir>/<stem>_foot_sticking.npz``: ``sticking`` (T, 2) bool for xy-stick, plus
``windows_left`` / ``windows_right`` (n, 5) float arrays.
"""

from __future__ import annotations

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


def compute(src: np.ndarray, boxes: np.ndarray, flight_tol: float) -> tuple[np.ndarray, list, list]:
    """Stance masks and per-foot anchor windows for one source clip.

    Args:
        src: ``(T, J, 3)`` g1fk keypoints at the solver rate.
        boxes: ``(N, 6)`` court cuboids as center xyz + size xyz.
        flight_tol: Lowest toe clearance (m) above which a frame may be labelled flight.

    Returns:
        ``(T, 2)`` stance masks [left, right] and each foot's ``[start, end, x, y, z]`` windows.
    """
    toe_idx = [G1FK_DEMO_JOINTS.index(n) for n in TOE_NAMES_BY_FORMAT["g1fk"]]
    toes = src[:, toe_idx]  # (T, 2, 3)
    surf = np.stack([terrain_z(toes[:, k, :2], boxes) for k in range(2)], axis=1)  # (T, 2)
    clear = toes[:, :, 2] - surf
    speed = np.zeros_like(clear)
    speed[1:] = np.linalg.norm(np.diff(toes[:, :, :2], axis=0), axis=2)

    # Global support-state labeling (Viterbi over {both, left, right, flight}): per-frame thresholds
    # flash in and out on dirty sources, snapping feet to new anchors at every flicker. Emissions are
    # POSITION-based (height above the terrain under the foot -- link velocity is contaminated by base
    # drift and only vetoes at coarse scale); the switching penalty makes uncertainty PERSIST the
    # current state instead of flipping it.
    t_n = len(toes)
    contact_cost = np.zeros((t_n, 2))
    swing_cost = np.zeros((t_n, 2))
    for k in range(2):
        c = clear[:, k]
        contact_cost[:, k] = np.maximum(0.0, c - 0.05) * 20.0 + np.maximum(0.0, speed[:, k] * 30.0 - 0.45) * 2.0
        swing_cost[:, k] = np.maximum(0.0, 0.05 - c) * 20.0
    STATES = ((True, True), (True, False), (False, True), (False, False))  # both, L, R, flight
    emis = np.zeros((t_n, 4))
    for si, (lc, rc) in enumerate(STATES):
        emis[:, si] = (contact_cost[:, 0] if lc else swing_cost[:, 0]) + (
            contact_cost[:, 1] if rc else swing_cost[:, 1]
        )
    emis[:, 3] += np.maximum(0.0, flight_tol - clear.min(axis=1)) * 10.0  # flight needs BOTH feet high
    SWITCH = 1.2  # per-foot state change penalty: the persistence knob
    trans = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            trans[a, b] = SWITCH * sum(STATES[a][f] != STATES[b][f] for f in range(2))
    dp = np.full((t_n, 4), np.inf)
    bk = np.zeros((t_n, 4), dtype=int)
    dp[0] = emis[0]
    for t in range(1, t_n):
        tot = dp[t - 1][:, None] + trans
        bk[t] = tot.argmin(axis=0)
        dp[t] = tot.min(axis=0) + emis[t]
    path = np.zeros(t_n, dtype=int)
    path[-1] = int(dp[-1].argmin())
    for t in range(t_n - 2, -1, -1):
        path[t] = bk[t + 1, path[t + 1]]
    masks = np.array([[STATES[si][0], STATES[si][1]] for si in path], dtype=bool)

    # calibrated toe-center offset above the surface when planted (per clip; sphere radius + skin)
    planted = clear[masks]
    offset = (
        float(np.clip(np.percentile(planted, 20), 0.008, 0.05)) if planted.size else 0.02
    )  # floor: sphere r=5mm + margin

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
                e_land = min(e, t + max(3, (e - t + 1) // 3))  # landing portion: before source drift accumulates
                x_a = float(np.median(toes[t : e_land + 1, k, 0]))
                y_a = float(np.median(toes[t : e_land + 1, k, 1]))
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
    ap.add_argument("--flight_tol", type=float, default=0.15, help="Min best-toe clearance (m) for true flight.")
    args = ap.parse_args()

    court = json.loads(Path(args.courts_json).read_text())["courts"][args.court]
    boxes = np.array([[*p["pos"], *p["size"]] for p in court["prims"]], dtype=np.float64)
    for d in args.seq_dirs:
        seq = Path(d)
        src = np.load(seq / f"{seq.name}.npy")[::DOWNSAMPLE]
        masks, wl, wr = compute(src, boxes, args.flight_tol)
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
