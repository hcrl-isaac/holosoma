"""Anatomical joint angles from source keypoint positions, as targets for the retarget solve.

Position matching alone does not preserve joint angles when the robot's proportions differ from the
human's, and for expressive motion the angles are the content. The 1-DOF hinges (elbow flexion, knee
flexion) have an unambiguous angle in the source: the angle between the two segments they join.
"""

from __future__ import annotations

import numpy as np

# SMPL body joint indices
SMPL = {"L_Sho": 16, "L_Elb": 18, "L_Wri": 20, "R_Sho": 17, "R_Elb": 19, "R_Wri": 21,
        "L_Hip": 1, "L_Kne": 4, "L_Ank": 7, "R_Hip": 2, "R_Kne": 5, "R_Ank": 8}


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


def t1_joint_angle_targets(joints: np.ndarray) -> dict[str, np.ndarray]:
    """Target angles for T1's flexion hinges, signed to match each joint's own range.

    Args:
        joints: ``(T, J, 3)`` source joint positions, any consistent scale.

    Returns:
        Mapping of T1 joint name to a ``(T,)`` target angle track.
    """
    s = SMPL
    l_elb = _bend(joints, s["L_Sho"], s["L_Elb"], s["L_Wri"])
    r_elb = _bend(joints, s["R_Sho"], s["R_Elb"], s["R_Wri"])
    l_kne = _bend(joints, s["L_Hip"], s["L_Kne"], s["L_Ank"])
    r_kne = _bend(joints, s["R_Hip"], s["R_Kne"], s["R_Ank"])
    # Elbow_Yaw is the flexion hinge and its range is one-sided: [-2.44, 0] left, [0, 2.44] right.
    # Knee_Pitch flexes positive on both sides.
    return {
        "Left_Elbow_Yaw": -l_elb,
        "Right_Elbow_Yaw": r_elb,
        "Left_Knee_Pitch": l_kne,
        "Right_Knee_Pitch": r_kne,
    }
