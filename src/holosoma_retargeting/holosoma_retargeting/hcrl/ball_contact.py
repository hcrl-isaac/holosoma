"""Fixed-size ball geometry for retargeting a scaled human against an object that does not scale.

Retargeting shrinks the human to robot height, so every distance in the motion shrinks with it -- but
a real ball keeps its radius. A foot that was tangent to the ball for the human therefore ends up
``(1 - scale) * radius`` INSIDE it for the robot, and the kick geometry is wrong. The retargeter
takes ball centres plus these foot points and keeps the foot out of the sphere.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Soccer-X's own ball, measured from the resting centre height of slow ground-level frames
# (0.1004 +- 0.0089 m). The source sidecar's clearances are measured against it, so changing it means
# regenerating them (``soccerx_source``) as well as re-solving.
BALL_RADIUS_M = 0.10

# No ball travels this far in one 30 fps frame (45 m/s); a step past it is a dropout in the track,
# and a bogus centre that lands on a foot would shove it aside for no reason.
BALL_STEP_MAX_M = 1.5

# Cap on the clearance the robot is asked to reproduce. At 0 the target is pure non-penetration:
# a positive band would chase a ball track that is only registered to the mocap to within a few cm.
BALL_CLEARANCE_BAND_M = 0.0


def foot_surface_points(model: mujoco.MjModel, foot_links: dict[str, str], voxel_m: float = 0.01) -> dict:
    """Collision-mesh vertices of each foot, in that foot body's own frame, thinned to a voxel grid.

    Args:
        model: Mujoco model containing the foot bodies.
        foot_links: Per-side foot body name (``RobotConfig.FOOT_LINKS``).
        voxel_m: Grid pitch for thinning; the ball-distance discretization error it costs is
            ``voxel_m ** 2 / (8 * radius)``, sub-mm at the default.

    Returns:
        Per-side arrays of shape (n, 3).
    """
    out = {}
    for side, link in foot_links.items():
        body_id = model.body(link).id
        meshes = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body_id or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[geom]
            adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
            verts = model.mesh_vert[adr : adr + num].astype(np.float64) @ rot.reshape(3, 3).T
            meshes.append(verts + model.geom_pos[geom])
        if not meshes:
            raise ValueError(f"foot body {link!r} has no collision mesh to build ball contact points from")
        verts = np.concatenate(meshes)
        _, keep = np.unique(np.round(verts / voxel_m).astype(np.int64), axis=0, return_index=True)
        out[side] = verts[np.sort(keep)]
    return out


def to_solver_frame(ball: np.ndarray, scale: float, radius: float = BALL_RADIUS_M) -> np.ndarray:
    """Map source-frame ball centres into the frame the retargeting targets live in.

    Ground-plane distances scale with the human, so xy follows ``preprocess_motion_data``. Height
    does not: a resting ball sits at its own radius whatever the player's size, so only the clearance
    above that scales. Both are referenced to the source's own ground plane, which is where the sole
    terms park the robot -- not to the deeper ``z_min`` the keypoints are dropped by.

    Args:
        ball: Ball centres of shape (T, 3), in the source npz frame.
        scale: The ``smpl_scale`` applied to the keypoints.
        radius: Ball radius in metres.

    Returns:
        Ball centres of shape (T, 3) in the solver frame, NaN on frames the track does not cover.
    """
    out = scale * ball
    clearance = np.maximum(ball[:, 2] - radius, 0.0)  # a rigid ball cannot sink into a flat floor
    out[:, 2] = radius + scale * clearance
    step = np.zeros(len(ball))
    step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    jump = np.maximum(step, np.roll(step, -1)) > BALL_STEP_MAX_M  # drop both ends of a teleport
    out[jump | ~np.isfinite(ball).all(axis=1)] = np.nan
    return out


def target_clearance(ball_gap: np.ndarray, band: float = BALL_CLEARANCE_BAND_M) -> np.ndarray:
    """Clearance the robot's foot must keep from the ball surface, per frame and foot.

    The human's own clearance is an ABSOLUTE distance to an object that never shrank, so it carries
    across unscaled. Where the source itself has the ball inside the foot the target is negative:
    retargeting must not deepen a mocap registration error, but it cannot undo one either.

    Args:
        ball_gap: The human's foot-to-ball-surface distance, shape (T, 2).
        band: Cap on the clearance demanded.

    Returns:
        Array of shape (T, 2).
    """
    return np.minimum(ball_gap, band)


def ball_gaps(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    ball: np.ndarray,
    foot_links: dict[str, str],
    foot_points: dict,
    radius: float = BALL_RADIUS_M,
) -> np.ndarray:
    """FK the output and measure each foot surface's signed distance to the ball surface.

    Args:
        model: Mujoco model matching ``qpos``.
        qpos: Retargeted configuration, shape (T, nq).
        ball: Solver-frame ball centres, shape (T, 3); NaN rows are skipped.
        foot_links: Per-side foot body name (``RobotConfig.FOOT_LINKS``).
        foot_points: Per-side foot-frame surface points from :func:`foot_surface_points`.
        radius: Ball radius in metres.

    Returns:
        Array of shape (T, 2) for [left, right]; negative is penetration, NaN where no valid ball.
    """
    data = mujoco.MjData(model)
    sides = list(foot_links)
    ids = [model.body(foot_links[side]).id for side in sides]
    gaps = np.full((len(qpos), len(sides)), np.nan)
    for t, row in enumerate(qpos):
        centre = ball[min(t, len(ball) - 1)]
        if not np.isfinite(centre).all():
            continue
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for k, (side, body_id) in enumerate(zip(sides, ids, strict=True)):
            pts = data.xpos[body_id] + foot_points[side] @ data.xmat[body_id].reshape(3, 3).T
            gaps[t, k] = np.linalg.norm(pts - centre, axis=1).min() - radius
    return gaps


def ball_score(gaps: np.ndarray, targets: np.ndarray | None = None) -> dict[str, float]:
    """Summarize one clip's foot-versus-ball geometry.

    Args:
        gaps: Output of :func:`ball_gaps`, shape (T, 2).
        targets: Per-foot clearance targets from :func:`target_clearance`, shape (T, 2); without
            them the deficit is measured against plain non-penetration.

    Returns:
        Dict of metric name to value. ``ball_deficit_m`` is the shipping number -- how much closer
        to the ball than the human was the retarget ever puts a foot -- and must be ~0.
    """
    keep = np.isfinite(gaps).any(axis=1)
    if not keep.any():
        return dict.fromkeys(
            ("ball_frames", "ball_closest_gap_m", "ball_penetration_frac", "ball_deficit_m", "ball_deficit_frac"),
            float("nan"),
        ) | {"ball_frames": 0.0}
    per_frame = np.nanmin(gaps[keep], axis=1)
    want = np.zeros_like(gaps) if targets is None else np.asarray(targets, dtype=float)
    deficit = np.nanmax(np.maximum(want[keep] - gaps[keep], 0.0), axis=1)
    return {
        "ball_frames": float(keep.sum()),
        "ball_closest_gap_m": float(per_frame.min()),
        "ball_penetration_frac": float((per_frame < 0).mean()),
        "ball_deficit_m": float(deficit.max()),
        "ball_deficit_frac": float((deficit > 0.001).mean()),
    }
