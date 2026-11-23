"""Per-robot joint names for the posture priors and joint-angle targets the retarget solve adds."""

from __future__ import annotations

JOINT_ROLES: dict[str, dict[str, tuple[str, ...]]] = {
    "t1": {
        "waist": ("Waist",),
        "hip_yaw": ("Left_Hip_Yaw", "Right_Hip_Yaw"),
        "shoulder_pitch": ("Left_Shoulder_Pitch", "Right_Shoulder_Pitch"),
        "shoulders": ("Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Right_Shoulder_Pitch", "Right_Shoulder_Roll"),
        "twist": ("Left_Elbow_Pitch", "Right_Elbow_Pitch"),
        "elbow": ("Left_Elbow_Yaw", "Right_Elbow_Yaw"),
        "head": ("AAHead_yaw", "Head_pitch"),
        "wrist": (),
    },
    "t2": {
        "waist": ("waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"),
        "hip_yaw": ("left_hip_yaw_joint", "right_hip_yaw_joint"),
        "shoulder_pitch": ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
        "shoulders": (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
        ),
        "twist": ("left_elbow_pitch_joint", "right_elbow_pitch_joint"),
        "elbow": ("left_elbow_yaw_joint", "right_elbow_yaw_joint"),
        "head": ("aa_head_yaw_joint", "head_pitch_joint"),
        "wrist": tuple(f"{s}_wrist_{a}_joint" for s in ("left", "right") for a in ("pitch", "yaw", "roll")),
    },
}
"""Role -> joint names; ``elbow`` and ``head`` are (left, right) and (yaw, pitch). Elbow flexion is negative on the
left for both robots."""
