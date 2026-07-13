"""SOMA skeleton npz -> g1fk climbing seq dir: source npy + q0 + verified stance windows + terrain scene.

Replaces the shipped-IK G1 FK pseudo-source with the clean SOMA skeleton (world Z-up meters, 120 fps):
bones map onto the ``G1FK_DEMO_JOINTS`` keypoint set and everything (keypoints, heightmap terrain,
contact anchors) is scaled by ONE uniform factor ``robot_height / human_height`` about the world
origin, so the solve happens in a consistent similarity-transformed world. Stance windows come
straight from scenebot's verified contact intervals (``<stem>_contacts.json``) -- the Viterbi /
threshold contact detection that caused foot snapping on dirty sources is bypassed entirely.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import mujoco
import numpy as np

from holosoma_retargeting.config_types.data_type import G1FK_DEMO_JOINTS, TOE_NAMES_BY_FORMAT
from holosoma_retargeting.hcrl.courts_to_scene import DEFAULT_ROBOT_XML, build_court_files
from holosoma_retargeting.hcrl.csv_to_g1fk import qpos_row
from holosoma_retargeting.hcrl.stance_windows import DOWNSAMPLE, terrain_z

G1_HEIGHT = 1.32
TOE_ANCHOR_OFF = 0.01  # planted toe-sphere center above the plateau: r=5mm + penetration tol + margin

# G1FK keypoint -> SOMA bone. Follows holosoma's ("mocap", "g1") choices (ToeBase->toe sphere,
# Foot->ankle) except the hand: Middle1 (fist center) instead of Middle3 (distal over-extends reach).
SOMA_TO_G1FK = {
    "pelvis_contour_link": "Hips",
    "left_hip_pitch_link": "LeftLeg",
    "left_knee_link": "LeftShin",
    "left_ankle_intermediate_1_link": "LeftFoot",
    "left_ankle_roll_sphere_5_link": "LeftToeBase",
    "right_hip_pitch_link": "RightLeg",
    "right_knee_link": "RightShin",
    "right_ankle_intermediate_1_link": "RightFoot",
    "right_ankle_roll_sphere_5_link": "RightToeBase",
    "left_shoulder_roll_link": "LeftArm",
    "left_elbow_link": "LeftForeArm",
    "left_sphere_hand_link": "LeftHandMiddle1",
    "right_shoulder_roll_link": "RightArm",
    "right_elbow_link": "RightForeArm",
    "right_sphere_hand_link": "RightHandMiddle1",
}


def soma_keypoints(pos: np.ndarray, names: list[str], scale: float) -> np.ndarray:
    """(T, 78, 3) SOMA bone positions -> (T, 15, 3) scaled keypoints in G1FK_DEMO_JOINTS order."""
    idx = [names.index(SOMA_TO_G1FK[j]) for j in G1FK_DEMO_JOINTS]
    return (pos[:, idx].astype(np.float64) * scale).astype(np.float32)


def heightmap_to_prims(grid: np.ndarray, origin: np.ndarray, res: float, min_h: float = 0.02) -> list[dict]:
    """Heightmap -> ground-anchored cuboid prims via greedy per-height rect cover (hm_to_courts pattern)."""

    def rects(mask: np.ndarray):
        used = np.zeros_like(mask, bool)
        out = []
        rows, cols = mask.shape
        for r in range(rows):
            c = 0
            while c < cols:
                if mask[r, c] and not used[r, c]:
                    c1 = c
                    while c1 + 1 < cols and mask[r, c1 + 1] and not used[r, c1 + 1]:
                        c1 += 1
                    r1 = r
                    while r1 + 1 < rows and mask[r1 + 1, c : c1 + 1].all() and not used[r1 + 1, c : c1 + 1].any():
                        r1 += 1
                    used[r : r1 + 1, c : c1 + 1] = True
                    out.append((r, c, r1 - r + 1, c1 - c + 1))
                    c = c1 + 1
                else:
                    c += 1
        return out

    ox, oy = float(origin[0]), float(origin[1])
    prims = []
    for h in np.unique(grid):
        if h < min_h:  # ground stays the mujoco plane
            continue
        for r, c, nr, nc in rects(np.isclose(grid, h)):
            prims.append(
                {
                    "pos": [ox + (r + nr / 2) * res, oy + (c + nc / 2) * res, float(h) / 2],
                    "size": [nr * res, nc * res, float(h)],
                }
            )
    return prims


def contact_windows(contacts: dict, src: np.ndarray, scale: float, boxes: np.ndarray) -> tuple[np.ndarray, list, list]:
    """Verified 120 fps contact intervals -> solver-rate sticking mask (T30, 2) + per-foot anchor windows.

    Each interval pins the toe sphere to ONE fixed point: xy = window-median scaled source toe (the
    json 'point' is the sole contact centroid, not the toe), z = scaled snapped_h plateau + toe offset.
    Windows also carry the source-implied stance ATTITUDE (yaw, pitch of the window-median ankle->toe
    axis, pitch relative to the skeleton's flat-stance axis): steep-descent stances rest toe-down, so
    a terrain-flat orientation target would drag the anchored toe off the plateau.
    """
    toe_idx = [G1FK_DEMO_JOINTS.index(n) for n in TOE_NAMES_BY_FORMAT["g1fk"]]
    ankle_idx = [G1FK_DEMO_JOINTS.index(f"{s}_ankle_intermediate_1_link") for s in ("left", "right")]
    toes = src[:, toe_idx].astype(np.float64)
    ankles = src[:, ankle_idx].astype(np.float64)
    # flat-stance axis pitch: ankle sits (sole_offset.ankle - sole_offset.toe) above the toe on flat ground
    so = contacts.get("sole_offset", {"ankle": 0.065, "toe": 0.016})
    dz_flat = (float(so["ankle"]) - float(so["toe"])) * scale
    t120 = len(toes)
    t30 = ceil(t120 / DOWNSAMPLE)
    mask = np.zeros((t30, 2), dtype=bool)
    windows: list[list] = [[], []]
    for k, side in enumerate(("left", "right")):
        bone_len = float(np.median(np.linalg.norm(toes[:, k] - ankles[:, k], axis=1)))
        pitch_flat = float(np.arcsin(np.clip(dz_flat / bone_len, -1.0, 1.0)))
        for iv in contacts["feet"][side]:
            t0, t1 = int(iv["t0"]), min(int(iv["t1"]), t120)  # t1 exclusive
            start, end = ceil(t0 / DOWNSAMPLE), (t1 - 1) // DOWNSAMPLE
            if end < start:
                continue
            x, y = (float(np.median(toes[t0:t1, k, a])) for a in (0, 1))
            plateau = max(float(iv["snapped_h"]) * scale, 0.0)
            surf = float(terrain_z(np.array([[x, y]]), boxes)[0])
            if abs(surf - plateau) > 0.02:
                print(f"[soma] WARN {side} [{start},{end}]: terrain under anchor {surf:.3f} != plateau {plateau:.3f}")
            axis = np.median(toes[t0:t1, k] - ankles[t0:t1, k], axis=0)
            yaw = float(np.arctan2(axis[1], axis[0]))
            pitch = float(np.arctan2(-axis[2], np.hypot(axis[0], axis[1]))) - pitch_flat  # toe-down positive
            windows[k].append([start, end, x, y, plateau + TOE_ANCHOR_OFF, yaw, pitch])
            mask[start : end + 1, k] = True
    return mask, windows[0], windows[1]


def build_q0(
    model: mujoco.MjModel, pelvis_xy: np.ndarray, yaw: float, init_csv: Path | None, ground_z: float = 0.0
) -> np.ndarray:
    """Initial qpos: standing joints (+ root z) from a shipped csv row 0, root xy/yaw from SOMA frame 0.

    ``ground_z`` lifts the root by the terrain height under the start pelvis (clips starting on a
    box/stair top would otherwise spawn inside the geometry -- an infeasible first frame).
    """
    if init_csv is not None:
        q = qpos_row(model, init_csv, 0)
    else:
        q = np.zeros(model.nq)
        q[2] = 0.78
    q[:2] = pelvis_xy
    q[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    # FK the lowest foot sphere and set root z so the sole rests just above the local terrain --
    # exact regardless of the init csv's own world frame (its row-0 z may or may not include terrain)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    sph = [model.body(f"{s}_ankle_roll_sphere_{i}_link").id for s in ("left", "right") for i in range(1, 6)]
    lowest = min(data.xpos[i][2] for i in sph) - 0.005  # sphere radius
    q[2] += ground_z + 0.005 - lowest
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="SOMA skeleton npz (pos [T,78,3] @120fps, names, human_height).")
    ap.add_argument("--contacts", required=True, help="scenebot <stem>_contacts.json (verified stance intervals).")
    ap.add_argument("--heightmap", required=True, help="scenebot <stem>_heightmap.npz (grid/origin/resolution).")
    ap.add_argument("--stem", required=True, help="Output clip stem (seq dir name).")
    ap.add_argument("--out_root", required=True, help="Climbing data root; writes <out_root>/<stem>/.")
    ap.add_argument("--init_csv", default=None, help="Shipped BONES csv: row 0 supplies q0 joints + root z.")
    ap.add_argument("--scale", type=float, default=None, help="Override human->G1 scale (default 1.32/height).")
    ap.add_argument("--robot_xml", default=str(DEFAULT_ROBOT_XML))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    pos, names = d["pos"], [str(n) for n in d["names"]]
    scale = args.scale if args.scale is not None else G1_HEIGHT / float(d["human_height"])
    src = soma_keypoints(pos, names, scale)

    seq_dir = Path(args.out_root) / args.stem
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / f"{args.stem}.npy", src)

    hm = np.load(args.heightmap)
    prims = heightmap_to_prims(hm["grid"], hm["origin"], float(hm["resolution"]))
    for p in prims:  # uniform scale about the world origin, same as the source keypoints
        p["pos"] = [round(v * scale, 4) for v in p["pos"]]
        p["size"] = [round(v * scale, 4) for v in p["size"]]
    build_court_files({"prims": prims}, seq_dir, Path(args.robot_xml))
    boxes = np.array([[*p["pos"], *p["size"]] for p in prims], dtype=np.float64)

    contacts = json.loads(Path(args.contacts).read_text())
    toe_idx = [G1FK_DEMO_JOINTS.index(n) for n in TOE_NAMES_BY_FORMAT["g1fk"]]
    mask, wl, wr = contact_windows(contacts, src, scale, boxes)
    np.savez(
        seq_dir / f"{args.stem}_foot_sticking.npz",
        sticking=mask,
        toe_names=TOE_NAMES_BY_FORMAT["g1fk"],
        windows_left=np.array(wl, dtype=float).reshape(-1, 7),
        windows_right=np.array(wr, dtype=float).reshape(-1, 7),
    )

    model = mujoco.MjModel.from_xml_path(args.robot_xml)
    hips, ll, rl = (
        src[0, G1FK_DEMO_JOINTS.index(n)]
        for n in ("pelvis_contour_link", "left_hip_pitch_link", "right_hip_pitch_link")
    )
    lat = ll - rl
    fwd = np.cross([lat[0], lat[1], 0.0], [0.0, 0.0, 1.0])
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    ground_z = float(terrain_z(np.asarray(hips[:2], dtype=np.float64)[None], boxes)[0])
    np.save(
        seq_dir / f"{args.stem}_q0.npy",
        build_q0(model, hips[:2], yaw, Path(args.init_csv) if args.init_csv else None, ground_z),
    )

    toe_min = float(src[::DOWNSAMPLE][:, toe_idx, 2].min())  # preprocess_motion_data will shift by this
    print(
        f"[soma] {args.stem}: T={len(src)} @120fps scale={scale:.4f} | {len(prims)} terrain prims | "
        f"windows L{len(wl)}/R{len(wr)} | solver z-shift {toe_min * 100:.1f}cm | yaw0 {np.degrees(yaw):.0f}deg | "
        f"q0 ground_z {ground_z:.3f}"
    )
    for side, ws in (("L", wl), ("R", wr)):
        for s0, s1, x, y, z, yw, pt in ws:
            print(
                f"    {side} [{int(s0):3d},{int(s1):3d}] anchor ({x:+.3f}, {y:+.3f}, {z:.3f}) "
                f"yaw {np.degrees(yw):+.0f}deg pitch {np.degrees(pt):+.0f}deg"
            )


if __name__ == "__main__":
    main()
