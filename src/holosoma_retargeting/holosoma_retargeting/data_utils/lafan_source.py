"""LAFAN1 world joint positions -> holosoma's ``lafan`` source files, without the mirror image.

holosoma's lafan loader turns y-up into z-up by swapping y and z, a reflection, and its keypoint list names
the joints right-for-left to match, so every take comes out mirrored: the robot gestures with the other
hand. Written here in ``LAFAN_DEMO_JOINTS`` order by name with the forward axis negated, the loader's
transform lands on a proper rotation instead.

    python -m holosoma_retargeting.data_utils.lafan_source --bvh-dir <bvh> --positions-dir <npy> --out-dir <src> \
        sprint1_subject2 walk3_subject4:198:212
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS

FPS = 30.0
ALIASES = {"LeftToe": "LeftToeBase", "RightToe": "RightToeBase"}


def bvh_joint_names(bvh_path: Path) -> list[str]:
    """Joint names in the order the BVH declares them (its positions follow the same order)."""
    names = []
    for line in bvh_path.read_text().splitlines():
        words = line.split()
        if words and words[0] in ("ROOT", "JOINT"):
            names.append(ALIASES.get(words[1], words[1]))
    return names


def to_holosoma(
    positions: np.ndarray, names: list[str], start_s: float | None = None, end_s: float | None = None
) -> np.ndarray:
    """Reorder, un-mirror and crop one take.

    Args:
        positions: ``(T, J, 3)`` y-up world positions in meters, in ``names`` order.
        names: The BVH joint names.
        start_s: Window start in seconds, or None for the take's start.
        end_s: Window end in seconds, or None for the take's end.

    Returns:
        ``(T', len(LAFAN_DEMO_JOINTS), 3)`` positions for holosoma's lafan loader.
    """
    missing = [n for n in LAFAN_DEMO_JOINTS if n not in names]
    if missing:
        raise ValueError(f"BVH lacks joints {missing}")
    out = positions[:, [names.index(n) for n in LAFAN_DEMO_JOINTS]].copy()
    out[..., 2] *= -1.0
    a = 0 if start_s is None else round(start_s * FPS)
    b = len(out) if end_s is None else round(end_s * FPS)
    return out[a:b]


def head_angles(bvh_path: Path) -> np.ndarray:
    """The head's yaw and pitch relative to the chest, as T1's ``AAHead_yaw`` and ``Head_pitch`` targets.

    Needs the LAFAN1 repository's ``lafan1`` reader on ``PYTHONPATH``; it is not a declared dependency.

    Args:
        bvh_path: The take's BVH file.

    Returns:
        ``(T, 2)`` yaw (positive = turn left) and pitch (positive = nod down) in radians.
    """
    from lafan1 import extract, utils  # type: ignore[import-not-found]
    from scipy.spatial.transform import Rotation

    anim = extract.read_bvh(str(bvh_path))
    grot, gpos = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    names = [ALIASES.get(n, n) for n in anim.bones]
    idx = {n: names.index(n) for n in ("Hips", "LeftUpLeg", "LeftFoot", "LeftToeBase", "Spine2", "Head")}
    # anatomical axes from the rest skeleton (all local rotations zero): forward along the toes, up the spine
    rest = np.zeros_like(anim.offsets)
    for j, p in enumerate(anim.parents):
        rest[j] = anim.offsets[j] if p < 0 else rest[p] + anim.offsets[j]
    up = rest[idx["Head"]] - rest[idx["Hips"]]
    up /= np.linalg.norm(up)
    fwd = rest[idx["LeftToeBase"]] - rest[idx["LeftFoot"]]
    fwd -= np.dot(fwd, up) * up
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    if np.dot(rest[idx["LeftUpLeg"]] - rest[idx["Hips"]], left) <= 0:
        raise ValueError("the rest skeleton's left hip is not on its left; the BVH is mirrored")
    basis = np.stack([fwd, left, up], axis=1)
    rot = Rotation.from_quat(grot[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*grot.shape[:2], 3, 3)
    # the reader's quaternions must reproduce its own FK positions, or the layout above is wrong
    child, parent = idx["Head"], anim.parents[idx["Head"]]
    fk = gpos[:, parent] + np.einsum("tij,j->ti", rot[:, parent], anim.offsets[child])
    if not np.allclose(fk, gpos[:, child], atol=1e-3):
        raise ValueError("BVH quaternion layout does not reproduce the reader's FK")
    rel = np.einsum("tji,tjk->tik", rot[:, idx["Spine2"]], rot[:, idx["Head"]])
    anat = np.einsum("ji,tjk,kl->til", basis, rel, basis)
    yaw = np.arctan2(anat[:, 1, 0], anat[:, 0, 0])
    pitch = np.arcsin(np.clip(-anat[:, 2, 0], -1.0, 1.0))
    return np.stack([yaw, pitch], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--positions-dir", type=Path, required=True, help="extract_global_positions.py output")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("takes", nargs="+", help="take, or take:start_s:end_s for a window")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.takes:
        take, *window = spec.split(":")
        start_s, end_s = (float(window[0]), float(window[1])) if window else (None, None)
        names = bvh_joint_names(args.bvh_dir / f"{take}.bvh")
        positions = np.load(args.positions_dir / f"{take}.npy")
        stem = take if not window else f"{take}_{window[0]}-{window[1]}s"
        out = to_holosoma(positions, names, start_s, end_s)
        np.save(args.out_dir / f"{stem}.npy", out)
        a = 0 if start_s is None else round(start_s * FPS)
        head = head_angles(args.bvh_dir / f"{take}.bvh")[a : a + len(out)]
        np.save(args.out_dir / f"{stem}_head.npy", head)
        print(f"wrote {stem}.npy: {out.shape[0]} frames ({out.shape[0] / FPS:.1f} s), head yaw/pitch sidecar")


if __name__ == "__main__":
    main()
