"""Retarget qpos npz -> play_dataset csv layout, plus the joint-order file that pins column meaning.

``play_dataset`` reads ``[pos(3), quat_xyzw(4), joint_pos(n)]`` per row and maps the joint columns onto
the Articulation through ``--joint_order``, so the order written here and the order in that file must
come from the same model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def joint_names(model: mujoco.MjModel) -> list[str]:
    """List hinge joint names in qpos order, skipping the floating base.

    Args:
        model: Mujoco model whose qpos layout the retarget output follows.

    Returns:
        Joint names ordered by qpos address.
    """
    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    return names


def convert(qpos: np.ndarray) -> np.ndarray:
    """Reorder one clip's qpos into the csv layout.

    Args:
        qpos: Retargeted configuration, shape (T, 7 + n_joints), quaternion in wxyz.

    Returns:
        Array of shape (T, 7 + n_joints) with the quaternion in xyzw.
    """
    quat_wxyz = qpos[:, 3:7]
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)
    return np.concatenate([qpos[:, :3], quat_xyzw, qpos[:, 7:]], axis=1)


def main() -> None:
    """CLI: convert retarget npz files into play_dataset csvs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retarget-dir", type=Path, required=True, help="Directory of retarget .npz outputs.")
    parser.add_argument("--robot-xml", type=Path, required=True, help="Mujoco xml the retarget was solved on.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write csvs into.")
    parser.add_argument("--clips", type=Path, default=None, help="Optional file listing clip stems, one per line.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    names = joint_names(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "joint_order.txt").write_text("\n".join(names) + "\n")

    stems = args.clips.read_text().split() if args.clips else [p.stem for p in sorted(args.retarget_dir.glob("*.npz"))]
    header = ",".join(["x", "y", "z", "qx", "qy", "qz", "qw"] + names)
    for stem in stems:
        qpos = np.load(args.retarget_dir / f"{stem}.npz")["qpos"]
        if qpos.shape[1] != 7 + len(names):
            raise ValueError(f"{stem}: qpos has {qpos.shape[1]} columns, expected {7 + len(names)}")
        np.savetxt(args.out_dir / f"{stem}.csv", convert(qpos), delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[qpos_to_csv] wrote {len(stems)} csvs + joint_order.txt ({len(names)} joints) to {args.out_dir}")


if __name__ == "__main__":
    main()
