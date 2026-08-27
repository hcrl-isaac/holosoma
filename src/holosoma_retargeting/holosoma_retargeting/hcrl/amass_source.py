"""AMASS / OMOMO npz -> smplx-format retargeting source npz.

Both corpora store SMPL(-H/-X) pose parameters rather than joint positions, so joints come from
:mod:`holosoma_retargeting.hcrl.smpl_fk`, exactly as the Soccer-X adapter does. The output layout is
identical to that adapter's minus the ball sidecar, so the same ``robot_retarget.py --data-format smplx
--task-type robot_only`` invocation consumes it.

Source frame rates vary per sequence (``mocap_framerate``), so clips are resampled to one target rate
before the solve; the retargeter has no notion of a per-clip rate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np

from holosoma_retargeting.hcrl.smpl_fk import (
    SMPL_BODY_JOINTS,
    load_smpl_model,
    plane_normals,
    skin_vertices,
    smpl_joint_positions,
    sole_vertices,
    to_z_up,
)

# SMPL takes 24 joints x 3; AMASS "SMPL+H G" stores 52 (root + 21 body + 30 hand) and OMOMO's SMPL-X 55.
# The leading 22 joints are shared, and the two SMPL hand joints are left at rest.
SMPL_POSE_DIM = 24 * 3
SHARED_POSE_DIM = 22 * 3

# AMASS ships only gendered fits ("SMPL+H G") for the full corpus, so FK-ing every clip through one body
# would mis-proportion half of them -- and the sole heights this adapter derives are what pins the feet to
# the floor. Each clip is FK'd on the model its fit used, named by the npz's own `gender` field.
SMPL_MODEL_FILES = {"male": "SMPL_MALE.pkl", "female": "SMPL_FEMALE.pkl", "neutral": "SMPL_NEUTRAL.pkl"}

# AMASS is not uniform across sub-datasets: the SMPL-X releases spell the frame rate differently and some
# store the pose split by body part instead of one array. Both differences are silent if unhandled -- a
# missed frame rate plays a 120 fps clip as 30, and the retarget comes out in slow motion.
FRAME_RATE_KEYS = ("mocap_framerate", "mocap_frame_rate", "frame_rate", "fps")
SPLIT_POSE_KEYS = ("root_orient", "pose_body")


def _resample(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Nearest-frame resample along axis 0. Nearest, not interpolated: SMPL poses are axis-angle, and
    interpolating them componentwise is wrong near the +/-pi wrap."""
    if abs(source_fps - target_fps) < 1e-6:
        return values
    count = max(1, round(values.shape[0] * target_fps / source_fps))
    idx = np.clip(np.round(np.arange(count) * source_fps / target_fps).astype(int), 0, values.shape[0] - 1)
    return values[idx]


def _gender(raw: dict) -> str:
    """The clip's fitted gender, normalized. Stored as bytes, a 0-d array, or a plain string."""
    value = raw.get("gender", "neutral")
    value = value.item() if isinstance(value, np.ndarray) else value
    value = value.decode() if isinstance(value, bytes) else str(value)
    value = value.strip().lower()
    return value if value in SMPL_MODEL_FILES else "neutral"


def load_models(model_dir: Path) -> dict[str, dict]:
    """Load the male/female/neutral SMPL models a gendered corpus needs, keyed by gender."""
    models = {}
    for gender, filename in SMPL_MODEL_FILES.items():
        path = model_dir / filename
        if path.exists():
            models[gender] = load_smpl_model(path)
    if "neutral" not in models:
        raise FileNotFoundError(f"{model_dir} has no {SMPL_MODEL_FILES['neutral']} to fall back on")
    return models


def _source_poses(raw: dict) -> np.ndarray:
    """The root+body pose block, from either a single ``poses`` array or split per-part fields."""
    if "poses" in raw:
        return np.asarray(raw["poses"], dtype=np.float64)
    if all(key in raw for key in SPLIT_POSE_KEYS):
        parts = [np.asarray(raw[key], dtype=np.float64) for key in SPLIT_POSE_KEYS]
        return np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=-1)
    raise ValueError(f"no pose data: expected 'poses' or {SPLIT_POSE_KEYS}, found {sorted(raw)}")


def _frame_rate(raw: dict) -> float:
    """The clip's source frame rate. Raises rather than guessing: a wrong rate is silent slow motion."""
    for key in FRAME_RATE_KEYS:
        if key in raw:
            return float(np.asarray(raw[key]).item())
    raise ValueError(f"no frame rate: expected one of {FRAME_RATE_KEYS}, found {sorted(raw)}")


def _body_pose(poses: np.ndarray) -> np.ndarray:
    """Take the shared root+body block of an SMPL-H/-X pose array into SMPL's 24-joint layout."""
    if poses.shape[-1] < SHARED_POSE_DIM:
        raise ValueError(f"pose array has {poses.shape[-1]} values per frame, expected at least {SHARED_POSE_DIM}")
    out = np.zeros((poses.shape[0], SMPL_POSE_DIM), dtype=np.float64)
    out[:, :SHARED_POSE_DIM] = poses[:, :SHARED_POSE_DIM]
    return out


# Signed volume of three anatomically independent body axes. Its SIGN is a handedness invariant: a
# frame conversion that mirrors the source flips it, silently swapping left and right limbs while every
# position and height check still passes. MEASURED, not assumed: 300/300 of the validated Soccer-X
# sources give a positive value (median 0.0082).
CHIRALITY_REFERENCE_SIGN = 1.0


def up_axis(joints: np.ndarray) -> int:
    """Index of the axis the body stands along, from the mean head-minus-feet vector.

    SMPL-native sources (AMASS) are y-up and need rotating; OMOMO stores z-up already. Rotating a z-up
    source lays the person on their side, and NO handedness or height check catches it -- a rotation
    preserves chirality, and the "ground" offset just re-centres the wreckage.

    Args:
        joints: SMPL body joint positions, shape ``(frames, >=16, 3)``.

    Returns:
        The axis index (2 for z-up), i.e. where head-minus-feet is largest and positive.
    """
    head_minus_feet = (joints[:, 15] - joints[:, [10, 11]].mean(axis=1)).mean(axis=0)
    return int(np.argmax(head_minus_feet))


def chirality(joints: np.ndarray) -> float:
    """Median signed volume of (hip axis, foot-forward axis, spine axis) over the clip.

    The three axes must be anatomically independent -- deriving one as a cross product of the other two
    makes the determinant unconditionally positive and the test vacuous.

    Args:
        joints: SMPL body joint positions, shape ``(frames, >=16, 3)``.

    Returns:
        The median determinant. Compare its SIGN with :data:`CHIRALITY_REFERENCE_SIGN`; the magnitude is
        just body scale.
    """
    across = joints[:, 2] - joints[:, 1]  # left hip -> right hip
    up = joints[:, 15] - joints[:, 0]  # pelvis -> head
    forward = joints[:, [10, 11]].mean(axis=1) - joints[:, [7, 8]].mean(axis=1)  # ankles -> toes
    return float(np.median(np.linalg.det(np.stack([across, forward, up], axis=1))))


def convert_clip(
    models: dict[str, dict],
    soles: dict[str, dict[str, np.ndarray]],
    raw: dict,
    out_dir: Path,
    target_fps: float,
    name: str,
    source_fps: float | None = None,
) -> dict:
    """Convert one AMASS/OMOMO npz into a retargeting source npz plus a metadata sidecar.

    Args:
        models: Gender-keyed SMPL models from :func:`load_models`.
        soles: Gender-keyed sole vertex indices from :func:`sole_vertices`.
        raw: Mapping with ``trans`` and either ``poses`` or ``root_orient``/``pose_body``.
        out_dir: Directory to write ``<name>.npz`` and ``<name>.meta.json`` into.
        target_fps: Rate every clip is resampled to.
        name: Output stem; also the ``--task-name`` the retargeter is called with.
        source_fps: Rate override for corpora that carry none (OMOMO); ``None`` reads it from ``raw``.

    Returns:
        Metadata dict for this clip (name, gender, frames, fps, ground offset, chirality).
    """
    gender = _gender(raw)
    model, clip_soles = models.get(gender, models["neutral"]), soles.get(gender, soles["neutral"])
    source_fps = _frame_rate(raw) if source_fps is None else source_fps
    poses = _resample(_body_pose(_source_poses(raw)), source_fps, target_fps)
    trans = _resample(np.asarray(raw["trans"], dtype=np.float64), source_fps, target_fps)

    joints = smpl_joint_positions(model, poses, trans)
    # rotate only a y-up source; see up_axis
    rotate = up_axis(joints) != 2
    orient = to_z_up if rotate else (lambda p: p)
    joints = orient(joints)[:, :SMPL_BODY_JOINTS]
    if up_axis(joints) != 2:
        raise ValueError(f"body does not stand along z after conversion (up axis {up_axis(joints)})")
    sole_points = [orient(skin_vertices(model, poses, trans, clip_soles[side])) for side in ("left", "right")]
    sole_normal = np.stack([plane_normals(p) for p in sole_points], axis=1)
    sole_height = np.stack([p[..., 2].min(axis=1) for p in sole_points], axis=1)

    # Put the feet on z=0 so the retargeter's contact logic sees a floor; AMASS clips float arbitrarily.
    ground = float(np.median(np.min(joints[:, [10, 11], 2], axis=1)))
    joints[..., 2] -= ground
    sole_height -= float(np.median(np.min(sole_height, axis=1)))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{name}.npz",
        global_joint_positions=joints.astype(np.float32),
        height=model["height"],
        sole_normal=sole_normal.astype(np.float32),
        sole_height=sole_height.astype(np.float32),
    )
    meta = {
        "name": name,
        "gender": gender,
        "frames": int(joints.shape[0]),
        "source_fps": source_fps,
        "rotated_to_z_up": rotate,
        "fps": target_fps,
        "ground": ground,
        "chirality": chirality(joints),
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta))
    return meta


def iter_clips(root: Path) -> Iterator[tuple[str, dict]]:
    """Yield ``(name, raw)`` for every sequence under ``root``.

    Handles both corpus shapes: an AMASS tree of per-sequence ``.npz`` (the dataset/subject/sequence
    path becomes the name), or an OMOMO ``.p`` joblib dict of sequences keyed by index.
    """
    if root.is_file():
        for seq in joblib.load(root).values():
            yield str(seq["seq_name"]), seq
        return
    for path in sorted(p for p in root.rglob("*.npz") if not p.name.startswith("shape")):
        yield "_".join(path.relative_to(root).with_suffix("").parts), np.load(path, allow_pickle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AMASS/OMOMO npz clips into retargeting sources.")
    parser.add_argument(
        "--clip-root",
        type=Path,
        required=True,
        help="AMASS root dir searched recursively for *.npz, or an OMOMO *.p sequence file.",
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        required=True,
        help="Directory holding SMPL_{MALE,FEMALE,NEUTRAL}.pkl; the clip's own gender field picks one.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for source npz files.")
    parser.add_argument("--fps", type=float, default=30.0, help="Rate every clip is resampled to.")
    parser.add_argument("--min-frames", type=int, default=30, help="Skip clips shorter than this after resampling.")
    parser.add_argument(
        "--assume-fps",
        type=float,
        default=None,
        help="Source rate for corpora that store none (OMOMO is 30). Only set it when you know the rate: "
        "a wrong value retargets the whole corpus at the wrong speed.",
    )
    args = parser.parse_args()

    models = load_models(args.smpl_model_dir)
    soles = {gender: sole_vertices(model) for gender, model in models.items()}
    print(f"[amass] body models: {sorted(models)}")
    written, skipped, rates = [], 0, {}
    for name, raw in iter_clips(args.clip_root):
        try:
            meta = convert_clip(models, soles, raw, args.out_dir, args.fps, name, source_fps=args.assume_fps)
        except (KeyError, ValueError) as err:
            print(f"[amass] SKIP {name}: {err}")
            skipped += 1
            continue
        if meta["frames"] < args.min_frames:
            skipped += 1
            continue
        rates[meta["source_fps"]] = rates.get(meta["source_fps"], 0) + 1
        written.append(meta)

    (args.out_dir / "clips.txt").write_text("\n".join(m["name"] for m in written) + "\n")
    flipped = [m["name"] for m in written if m["chirality"] * CHIRALITY_REFERENCE_SIGN < 0]
    genders = {g: sum(m["gender"] == g for m in written) for g in SMPL_MODEL_FILES}
    print(f"[amass] wrote {len(written)} clips, skipped {skipped}; by gender {genders}")
    print(f"[amass] source frame rates seen: {dict(sorted(rates.items()))}")
    rotated = sum(m["rotated_to_z_up"] for m in written)
    print(f"[amass] rotated y-up -> z-up: {rotated}/{len(written)} clips (AMASS expects all, OMOMO none)")
    if flipped:
        print(f"\033[91m[amass] {len(flipped)} clips are MIRRORED (left/right swapped): {flipped[:5]}\033[0m")


if __name__ == "__main__":
    main()
