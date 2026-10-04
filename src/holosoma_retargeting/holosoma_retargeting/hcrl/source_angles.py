"""Anatomical joint angles from source keypoint positions, as targets for the retarget solve.

Position matching alone does not preserve joint angles when the robot's proportions differ from the
human's, and for expressive motion the angles are the content. The 1-DOF hinges (elbow flexion, knee
flexion) have an unambiguous angle in the source: the angle between the two segments they join.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# each T1 hinge's (proximal, hinge, distal) source joints under their SMPL and LAFAN/mocap names, and the
# sign that maps the interior bend onto the joint's range (Elbow_Yaw is one-sided, negative on the left)
T1_HINGES = {
    "Left_Elbow_Yaw": ((("L_Shoulder", "LeftArm"), ("L_Elbow", "LeftForeArm"), ("L_Wrist", "LeftHand")), -1.0),
    "Right_Elbow_Yaw": ((("R_Shoulder", "RightArm"), ("R_Elbow", "RightForeArm"), ("R_Wrist", "RightHand")), 1.0),
    "Left_Knee_Pitch": ((("L_Hip", "LeftUpLeg"), ("L_Knee", "LeftLeg"), ("L_Ankle", "LeftFoot")), 1.0),
    "Right_Knee_Pitch": ((("R_Hip", "RightUpLeg"), ("R_Knee", "RightLeg"), ("R_Ankle", "RightFoot")), 1.0),
}


def _bend(p: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Interior bend at ``b`` between segments ``a->b`` and ``b->c``.

    Args:
        p: ``(T, J, 3)`` joint positions.
        a: Proximal joint index.
        b: The hinge joint index.
        c: Distal joint index.

    Returns:
        ``(T,)`` bend angle in radians; 0 when the limb is straight.
    """
    u, v = p[:, b] - p[:, a], p[:, c] - p[:, b]
    nu = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cos = np.divide(np.einsum("ij,ij->i", u, v), nu, out=np.zeros(len(p)), where=nu > 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def t1_joint_angle_targets(joints: np.ndarray, demo_joints: Sequence[str]) -> dict[str, np.ndarray]:
    """Target angles for T1's flexion hinges whose three source joints ``demo_joints`` names.

    Args:
        joints: ``(T, J, 3)`` source joint positions, any consistent scale.
        demo_joints: Name of each of the ``J`` source joints.

    Returns:
        Mapping of T1 joint name to a ``(T,)`` target angle track; a hinge the source lacks is absent.
    """
    index = {name: i for i, name in enumerate(demo_joints)}
    targets = {}
    for t1_joint, (chain, sign) in T1_HINGES.items():
        found = [next((index[n] for n in spellings if n in index), None) for spellings in chain]
        if None not in found:
            targets[t1_joint] = sign * _bend(joints, *found)
    return targets
