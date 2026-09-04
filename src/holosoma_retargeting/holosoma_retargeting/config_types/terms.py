"""Solver term weights and switches (the hcrl additions to the interaction-mesh retargeter)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal


@dataclass(frozen=True)
class SolverTerms:
    """Weights and switches for the retargeting objective, settable on the CLI as ``--terms.<name>``.

    Defaults are the plain solve: interaction mesh, smoothing, sole orientation, and the redundancy
    priors. Named presets (``--preset``) set a whole route at once.
    """

    # --- target preprocessing ---
    limb_retarget: bool = False
    """Rescale source keypoints to the robot's segment lengths (directions preserved)."""
    foot_calib: bool = False
    """Drop the targets so the planted source toe sits at the robot's toe height."""
    sole_planted: Literal["calib", "flat", "off"] = "calib"
    """Planted sole-normal handling: remove the planted median pitch, force flat, or leave the source."""
    toe_clamp: bool = True
    """Never command a toe target below the sole on flat ground."""
    foot_min_sep: float = 0.0
    """Widen foot targets laterally to at least this toe separation (m, 0 = off)."""

    # --- tracking terms ---
    laplacian_weight: float | None = None
    """Interaction-mesh weight (None = the solver's default)."""
    keypoint_weight: float = 0.0
    """Absolute position term on every mapped keypoint."""
    joint_angle_weight: float = 5.0
    """Track the source's elbow/knee flexion angles."""
    pelvis_weight: float = 5.0
    """Absolute position prior on the pelvis keypoint."""
    arm_weight: float = 2.0
    """Absolute position prior on the arm keypoints."""
    arm_plane_weight: float = 0.0
    """Match the shoulder-elbow-wrist plane normal to the source (fades out for straight arms)."""
    foot_yaw_weight: float = 2.0
    """Steer each foot's heading to the source ankle->toe heading."""
    sole_weight: float = 5.0
    """Match the sole plane orientation to the source."""
    sole_height_weight: float = 2000.0
    """Pull a planted sole down onto the floor (one-sided)."""
    root_rate_weight: float = 0.0
    """Match the source root angular rate."""

    # --- posture priors ---
    swing_ankle_weight: float = 0.5
    """Neutral-ankle prior while a foot swings."""
    joint_limit_weight: float = 50.0
    """Barrier inside ``joint_limit_margin`` of an actuated stop."""
    joint_limit_margin: float = 0.10
    joint_limit_margin_frac: float = 0.15
    joint_limit_joints: str = ""
    """Comma-separated joints the barrier is restricted to (empty = all)."""
    t1_manual: bool = False
    """T1 posture costs and range caps (the G1 config's hand-tuned regularizers, translated)."""
    waist_cost: float = 0.2
    hip_yaw_cost: float = 0.0
    shoulder_cost: float = 0.0
    twist_cost: float = 0.0
    elbow_cap: float = 1.6
    """Elbow flexion cap (rad) under ``t1_manual``; the joint range is 2.44."""
    straight_twist_weight: float = 0.0
    """Posture cost on the upper-arm twist while the source elbow is nearly straight."""

    # --- temporal smoothing ---
    smooth_weight: float | None = None
    """Scalar velocity smoothing on all rows (None = the solver's default)."""
    root_smooth: float = 8.0
    """Velocity smoothing on the root orientation rows."""
    accel_damp: float = 3.0
    """Acceleration damping on all rows."""
    joint_smooth: float = 0.0
    """Velocity smoothing on every actuated joint (per-joint weights below override)."""
    twist_smooth: float = 0.0
    shoulder_smooth: float = 0.0
    body_contact_gain: float = 0.0
    """Smoothing and damping x (1 + gain x non-foot bodies on the ground)."""
    body_contact_root: float = 0.0
    """Root-orientation smoothing added per non-foot body on the ground."""

    # --- contact handling ---
    foot_sticking: bool = True
    stick_band: bool = True
    """Stance band follows the source toe's own per-frame travel instead of a fixed 1 mm."""
    teleport_guard: bool = True
    self_collision: str = ""
    """Body pairs kept apart, e.g. ``left_foot_link:right_foot_link,Shank_Left:Shank_Right``."""
    self_collision_tol: float = 0.01
    self_collision_margin: float = 0.0
    """Soft repulsion starts inside this distance (m, 0 = off)."""
    self_collision_margin_weight: float = 100.0
    self_collision_escape: float = 0.02
    """Metres a violated pair may separate per SQP iteration."""
    foot_stack_clearance: float = 0.0
    """Extra vertical gap a crossing foot keeps over the stance foot (m, 0 = off)."""
    foot_stack_weight: float = 100.0
    ground_margin: float = 0.0
    """Soft cushion above the ground for non-foot bodies (m, 0 = off)."""
    ground_margin_weight: float = 200.0

    # --- ball (Soccer-X) ---
    ball_weight: float = 2000.0
    ball_band: float | None = None
    ball_constraint: bool = False

    # --- solve / debug ---
    n_iter: int = 0
    """SQP iterations per frame (0 = the solver's default)."""
    step_size: float = 0.0
    """Trust region per SQP iteration (0 = the solver's default)."""
    debug_terms: bool = False
    dump_targets: str = ""
    """Path to save the solver's per-frame targets to."""


def resolve_terms(cli: SolverTerms, preset: str | None) -> SolverTerms:
    """Preset values, with any field the CLI set away from its default winning.

    Args:
        cli: Terms as parsed from the command line.
        preset: Preset name in ``config_values.presets.PRESETS``, or None.

    Returns:
        The effective terms.
    """
    if preset is None:
        return cli
    from holosoma_retargeting.config_values.presets import PRESETS

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; available: {sorted(PRESETS)}")
    defaults = SolverTerms()
    merged = dict(PRESETS[preset])
    for f in fields(SolverTerms):
        v = getattr(cli, f.name)
        if v != getattr(defaults, f.name):
            merged[f.name] = v
    return SolverTerms(**merged)
