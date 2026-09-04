"""Named solver-term presets: the two validated T1 routes."""

from __future__ import annotations

_T1_COMMON = dict(
    limb_retarget=True,
    foot_calib=True,
    sole_height_weight=0.0,
    self_collision="left_foot_link:right_foot_link",
    self_collision_margin=0.04,
    self_collision_margin_weight=1000.0,
    self_collision_escape=0.005,
    foot_stack_clearance=0.03,
    foot_stack_weight=2000.0,
    t1_manual=True,
    hip_yaw_cost=0.5,
    shoulder_cost=0.5,
    twist_cost=0.0,
    straight_twist_weight=2.0,
    twist_smooth=5.0,
    shoulder_smooth=5.0,
    joint_smooth=2.0,
)

PRESETS: dict[str, dict] = {
    # keypoint + joint-angle tracking on top of the mesh; for non-manipulation corpora (AMASS)
    "t1_keypoint": dict(
        _T1_COMMON,
        keypoint_weight=50.0,
        joint_angle_weight=20.0,
        root_smooth=20.0,
        accel_damp=3.0,
        elbow_cap=2.44,
        body_contact_gain=1.5,
    ),
    # interaction mesh only, with posture priors; for manipulation corpora (OMOMO)
    "t1_laplacian": dict(
        _T1_COMMON,
        keypoint_weight=0.0,
        joint_angle_weight=0.0,
        joint_limit_weight=0.0,
        pelvis_weight=0.0,
        arm_weight=0.0,
        swing_ankle_weight=0.0,
        root_smooth=0.0,
        accel_damp=0.0,
        arm_plane_weight=0.5,
        body_contact_root=60.0,
    ),
}
