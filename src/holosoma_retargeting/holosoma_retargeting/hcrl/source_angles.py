"""Anatomical joint angles from source keypoint positions, as targets for the retarget solve.

Position matching alone does not preserve joint angles when the robot's proportions differ from the
human's, and for expressive motion the angles are the content. The 1-DOF hinges (elbow flexion, knee
flexion) have an unambiguous angle in the source: the angle between the two segments they join.
"""

from __future__ import annotations

import numpy as np

from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS, SMPLH_DEMO_JOINTS, SMPLX_DEMO_JOINTS

_SMPL_NAMES = (
    ("L_Sho", "L_Shoulder"),
    ("L_Elb", "L_Elbow"),
    ("L_Wri", "L_Wrist"),
    ("R_Sho", "R_Shoulder"),
    ("R_Elb", "R_Elbow"),
    ("R_Wri", "R_Wrist"),
    ("L_Hip", "L_Hip"),
    ("L_Kne", "L_Knee"),
    ("L_Ank", "L_Ankle"),
    ("R_Hip", "R_Hip"),
    ("R_Kne", "R_Knee"),
    ("R_Ank", "R_Ankle"),
)
_LAFAN_NAMES = (
    ("L_Sho", "LeftArm"),
    ("L_Elb", "LeftForeArm"),
    ("L_Wri", "LeftHand"),
    ("R_Sho", "RightArm"),
    ("R_Elb", "RightForeArm"),
    ("R_Wri", "RightHand"),
    ("L_Hip", "LeftUpLeg"),
    ("L_Kne", "LeftLeg"),
    ("L_Ank", "LeftFoot"),
    ("R_Hip", "RightUpLeg"),
    ("R_Kne", "RightLeg"),
    ("R_Ank", "RightFoot"),
)
# each format's keypoint indices by joint name: smplh (InterMimic) and smplx order their joints differently
SKELETONS = {
    fmt: {key: joints.index(name) for key, name in names}
    for fmt, joints, names in (
        ("smplh", SMPLH_DEMO_JOINTS, _SMPL_NAMES),
        ("smplx", SMPLX_DEMO_JOINTS, _SMPL_NAMES),
        ("lafan", LAFAN_DEMO_JOINTS, _LAFAN_NAMES),
    )
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


def t1_joint_angle_targets(joints: np.ndarray, data_format: str = "smplx") -> dict[str, np.ndarray]:
    """Target angles for T1's flexion hinges, signed to match each joint's own range.

    Args:
        joints: ``(T, J, 3)`` source joint positions, any consistent scale.
        data_format: Source skeleton, one of ``SKELETONS``.

    Returns:
        Mapping of T1 joint name to a ``(T,)`` target angle track.
    """
    if data_format not in SKELETONS:
        raise ValueError(f"no joint-angle map for data format {data_format!r}; known: {sorted(SKELETONS)}")
    s = SKELETONS[data_format]
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
