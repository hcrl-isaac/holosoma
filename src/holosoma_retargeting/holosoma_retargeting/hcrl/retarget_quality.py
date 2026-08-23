"""Robot- and format-agnostic quality metrics for a retarget output.

Reads a ``robot_retarget.py`` output npz (``qpos``, ``human_joints``) and scores it by mujoco-FK of
the OUTPUT, so the numbers reflect what the solve actually produced rather than what it was asked for.

TODO: the terrain model here is a flat floor at z=0. Clips retargeted against fitted courts (boxes,
stairs, edges) need the per-clip surface instead -- pass a ``terrain_z`` callable once the court
geometry is threaded through, and the stance/contact machinery in ``stance_windows`` generalized off
the ``g1fk`` format it is currently written against.
"""

from __future__ import annotations

import mujoco
import numpy as np

# A snap is a toe jump that is both absolutely large and much larger than the source's own step.
# The constrained solve trails the source by up to a frame through hard decelerations, so the source
# step is taken over a small window rather than the same frame.
SNAP_ABS_M = 0.08
SNAP_REL = 1.5
SNAP_LAG_FRAMES = 1

# A source step this large is only physical for a kick; isolated ones (no comparable neighbour) are
# mocap teleports in the source itself, which retargeting faithfully reproduces.
SOURCE_TELEPORT_M = 0.30
SOURCE_TELEPORT_ISOLATION = 0.15

# A source toe below this is treated as planted; the robot's sole must follow within the tolerance.
SOURCE_PLANTED_M = 0.03
SOLE_FOLLOW_TOL_M = 0.02


def _flat_ground(points_xy: np.ndarray) -> np.ndarray:
    """Surface height under each xy point; flat floor at z=0.

    Args:
        points_xy: Array of shape (..., 2).

    Returns:
        Zeros of shape ``points_xy.shape[:-1]``.
    """
    return np.zeros(points_xy.shape[:-1])


def sole_positions(model: mujoco.MjModel, qpos: np.ndarray, sole_links: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """FK the output qpos and read every sole sphere's world position.

    Args:
        model: Mujoco model matching ``qpos``.
        qpos: Array of shape (T, nq).
        sole_links: Per-side sole link names (``RobotConfig.SOLE_LINKS``).

    Returns:
        Per-side arrays of shape (T, n_spheres, 3).
    """
    data = mujoco.MjData(model)
    ids = {side: [model.body(name).id for name in names] for side, names in sole_links.items()}
    out = {side: np.zeros((len(qpos), len(body_ids), 3)) for side, body_ids in ids.items()}
    for t, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for side, body_ids in ids.items():
            out[side][t] = data.xpos[body_ids]
    return out


def score(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source_toes: np.ndarray,
    sole_links: dict[str, list[str]],
    fps: float,
    terrain_z=_flat_ground,
) -> dict[str, float]:
    """Score one retargeted clip for ground contact, penetration and teleporting feet.

    Args:
        model: Mujoco model matching ``qpos``.
        qpos: Retargeted configuration, shape (T, nq).
        source_toes: Source toe keypoints, shape (T, 2, 3), in the output's frame.
        sole_links: Per-side sole link names (``RobotConfig.SOLE_LINKS``).
        fps: Frame rate of ``qpos``, used to report the longest airborne stretch in seconds.
        terrain_z: Callable mapping (..., 2) xy points to surface height.

    Returns:
        Dict of metric name to value; ``snaps`` is the shipping gate and must be 0. ``support_coverage``
        is measured only over frames where the source itself is planted.
    """
    soles = sole_positions(model, qpos, sole_links)
    sides = list(sole_links)
    sole_z = np.stack(
        [(soles[side][..., 2] - terrain_z(soles[side][..., :2])).min(axis=1) for side in sides], axis=1
    ).min(axis=1)
    # Source keypoints are body joints, not soles: a planted SMPL foot still reads a few cm up, so
    # contact is judged against the source's own toe height rather than an absolute floor.
    src_z = source_toes[..., 2].min(axis=1)
    planted = src_z < SOURCE_PLANTED_M
    follows = sole_z < SOURCE_PLANTED_M + SOLE_FOLLOW_TOL_M

    airborne = ~planted
    longest, run = 0, 0
    for a in airborne:
        run = run + 1 if a else 0
        longest = max(longest, run)

    out_toe = np.stack([soles[side].mean(axis=1) for side in sides], axis=1)
    out_step = np.linalg.norm(np.diff(out_toe, axis=0), axis=-1)
    src_step = np.linalg.norm(np.diff(source_toes, axis=0), axis=-1)
    src_ref = src_step.copy()
    for shift in range(1, SNAP_LAG_FRAMES + 1):
        src_ref[shift:] = np.maximum(src_ref[shift:], src_step[:-shift])
        src_ref[:-shift] = np.maximum(src_ref[:-shift], src_step[shift:])
    snaps = (out_step > SNAP_ABS_M) & (out_step > SNAP_REL * src_ref)

    return {
        "frames": float(len(qpos)),
        "support_coverage": float(follows[planted].mean()) if planted.any() else float("nan"),
        "planted_frac": float(planted.mean()),
        "max_airborne_s": longest / float(fps),
        "foot_track_p95_m": float(np.percentile(np.abs(sole_z - src_z), 95)),
        "max_penetration_m": float(max(0.0, -sole_z.min())),
        "snaps": float(snaps.sum()),
        "max_step_m": float(out_step.max()) if out_step.size else 0.0,
        "max_src_step_m": float(src_step.max()) if src_step.size else 0.0,
        "source_teleports": float(source_teleports(source_toes)),
    }


def source_teleports(source_toes: np.ndarray) -> int:
    """Count isolated toe jumps in the SOURCE, which no retargeting setting can fix.

    Args:
        source_toes: Source toe keypoints, shape (T, 2, 3).

    Returns:
        Number of large steps whose neighbours are far smaller, i.e. mocap discontinuities rather
        than the ramped acceleration of a real kick.
    """
    step = np.linalg.norm(np.diff(source_toes, axis=0), axis=-1)
    count = 0
    for t, k in np.argwhere(step > SOURCE_TELEPORT_M):
        lo, hi = max(0, t - 2), min(len(step), t + 3)
        neighbours = np.concatenate([step[lo:t, k], step[t + 1 : hi, k]])
        if neighbours.size and neighbours.max() < SOURCE_TELEPORT_ISOLATION * step[t, k]:
            count += 1
    return count
