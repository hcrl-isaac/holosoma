"""Soccer-X clip .pt -> smplx-format retargeting source npz (+ ball/label sidecar).

Soccer-X stores SMPL pose parameters rather than joint positions, so joints are produced by
:mod:`holosoma_retargeting.hcrl.smpl_fk`. Its first 22 SMPL joints are exactly ``SMPLX_DEMO_JOINTS``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from holosoma_retargeting.hcrl.ball_contact import BALL_RADIUS_M
from holosoma_retargeting.hcrl.smpl_fk import (
    SMPL_BODY_JOINTS,
    foot_vertices,
    load_smpl_model,
    plane_normals,
    skin_vertices,
    smpl_joint_positions,
    sole_vertices,
    to_z_up,
)


def convert_clip(
    model: dict, clip_path: Path, out_dir: Path, soles: dict[str, np.ndarray], feet: dict[str, np.ndarray]
) -> dict:
    """Convert one Soccer-X clip into a retargeting source npz plus a metadata sidecar.

    Args:
        model: Output of :func:`load_smpl_model`.
        clip_path: Path to a Soccer-X ``.pt`` clip.
        out_dir: Directory to write ``<stem>.npz`` and ``<stem>.meta.json`` into.
        soles: Per-side sole vertex indices from :func:`sole_vertices`.
        feet: Per-side whole-foot vertex indices from :func:`foot_vertices`.

    Returns:
        Metadata dict for this clip (name, label, frames, ground offset).
    """
    clip = torch.load(clip_path, map_location="cpu", weights_only=False)
    poses, trans = clip["human_poses"].numpy(), clip["human_trans"].numpy()
    joints = smpl_joint_positions(model, poses, trans)
    joints = to_z_up(joints)[:, :SMPL_BODY_JOINTS]
    ball = to_z_up(clip["soccer_pos"].numpy().astype(np.float64))
    # Sole plane per foot: the retargeter matches the robot sole's normal to this, which is the only
    # thing that pins foot pitch/roll -- the joint mapping alone leaves them free.
    sole_points = [to_z_up(skin_vertices(model, poses, trans, soles[side])) for side in ("left", "right")]
    sole_normal = np.stack([plane_normals(p) for p in sole_points], axis=1)
    # lowest sole point per foot: matching the normal alone leaves the flattened foot hovering,
    # because rotating a toe-down sole flat lifts its lowest contact point
    sole_height = np.stack([p[..., 2].min(axis=1) for p in sole_points], axis=1)

    # Soccer-X is ground-aligned at z=0; drop any residual so the retargeter's contact logic sees a floor.
    ground = float(np.median(np.min(joints[:, [10, 11], 2], axis=1)))
    joints[..., 2] -= ground
    ball[..., 2] -= ground
    # The sole surface sits ~2 cm BELOW the toe joint, so it needs its own reference: measured against
    # the joint-derived ground a planted sole reads about -24 mm and would be driven into the floor.
    sole_height -= float(np.median(np.min(sole_height, axis=1)))

    # The human's own foot-to-ball clearance is what the robot has to reproduce in ABSOLUTE terms --
    # the ball does not shrink with the player, so the retargeter cannot infer it from scaled keypoints.
    foot_points = [to_z_up(skin_vertices(model, poses, trans, feet[side])) for side in ("left", "right")]
    ball_gap = np.stack(
        [np.linalg.norm(p - ball[:, None], axis=2).min(axis=1) - BALL_RADIUS_M for p in foot_points], axis=1
    )

    name = clip_path.stem
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
        "part": clip_path.parent.name,
        "label": clip["motion_label"],
        "motion_name": clip["motion_name"],
        # a clip's "frames" field can overstate its arrays (000917 declares 4699, stores 3519)
        "frames": int(joints.shape[0]),
        "declared_frames": int(clip["frames"]),
        "fps": 30,
        "ground_offset": ground,
    }
    np.savez(
        out_dir / f"{name}.ball.npz",
        soccer_pos=ball.astype(np.float32),
        soccer_ori=clip["soccer_ori"].numpy(),
        ball_gap=ball_gap.astype(np.float32),
    )
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta))
    return meta


def main() -> None:
    """CLI: convert a Soccer-X clip tree into retargeting source npz files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root", type=Path, required=True, help="Soccer-X 'clip' dir containing Part*/")
    parser.add_argument("--smpl-model", type=Path, required=True, help="Path to SMPL_MALE.pkl.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for source npz files.")
    parser.add_argument("--min-frames", type=int, default=0, help="Skip clips shorter than this many source frames.")
    args = parser.parse_args()

    model = load_smpl_model(args.smpl_model)
    soles, feet = sole_vertices(model), foot_vertices(model)
    clips = sorted(args.clip_root.glob("Part*/*.pt"))
    metas = []
    for i, clip in enumerate(clips):
        meta = convert_clip(model, clip, args.out_dir, soles, feet)
        if meta["frames"] < args.min_frames:
            for suffix in (".npz", ".ball.npz", ".meta.json"):
                (args.out_dir / f"{meta['name']}{suffix}").unlink()
            continue
        metas.append(meta)
        if (i + 1) % 200 == 0:
            print(f"[soccerx] {i + 1}/{len(clips)}")
    (args.out_dir / "manifest.json").write_text(json.dumps(metas, indent=1))
    print(f"[soccerx] wrote {len(metas)} source clips to {args.out_dir}")


if __name__ == "__main__":
    main()
