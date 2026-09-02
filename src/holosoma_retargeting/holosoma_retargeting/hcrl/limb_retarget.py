"""Rescale source keypoints so each limb segment matches the robot's own segment length.

A single uniform height scale preserves HUMAN proportions. When the robot is not human-proportioned
the mapped targets become unreachable, the IK saturates joints trying to span them, and the result is
the limbs-at-odd-angles signature. Rescaling segment by segment keeps every bone DIRECTION -- which is
what carries the pose -- while giving the solver targets the robot can actually reach.
"""

from __future__ import annotations

import numpy as np


def robot_segment_lengths(model, data, mapping: dict[str, str], mj, samples: int = 300):
    """Parent links, segment lengths, and whether each segment is rigid.

    A segment is RIGID when the mapped child's body is a direct child of the mapped parent's body: a urdf
    joint puts the child origin at a fixed offset, so that distance cannot change. Spanning two or more
    joints, the distance varies with the joints in between -- prescribing its zero-pose value pins those
    joints and the solver loses the limb. For those we record the MAXIMUM reachable distance instead, to
    be used as a cap rather than an equality.

    Args:
        model: MuJoCo model of the robot.
        data: MuJoCo data for ``model``.
        mapping: Source-joint name -> robot body name, in keypoint order.
        mj: The ``mujoco`` module.
        samples: Random configurations used to bound a non-rigid span.

    Returns:
        ``(parent, length, rigid)`` keyed by source-joint name.
    """
    import numpy as _np

    def fk_positions(q):
        data.qpos[:] = 0.0
        if q is not None:
            n = min(len(q), len(data.qpos))
            data.qpos[:n] = q[:n]
        if model.nq >= 7:  # a zero quaternion is invalid; keep the free joint upright
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(model, data)
        return {src: _np.array(data.xpos[bid]) for src, bid in body_of.items()}

    body_of = {}
    for src, body in mapping.items():
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise ValueError(f"mapped body {body!r} not in the robot model")
        body_of[src] = bid

    parent_body = {i: int(model.body_parentid[i]) for i in range(model.nbody)}
    mapped_by_bid = {bid: src for src, bid in body_of.items()}
    parent, rigid, hops = {}, {}, {}
    for src, bid in body_of.items():
        cur, found, n = parent_body[bid], None, 1
        while cur > 0:
            if cur in mapped_by_bid and mapped_by_bid[cur] != src:
                found = mapped_by_bid[cur]
                break
            cur = parent_body[cur]
            n += 1
        parent[src] = found
        hops[src] = n
        rigid[src] = found is not None and parent_body[bid] == body_of[found]

    pos0 = fk_positions(None)
    length = {}
    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    rng = _np.random.default_rng(0)
    for src, p_src in parent.items():
        if p_src is None:
            length[src] = 0.0
            continue
        if rigid[src]:
            length[src] = float(_np.linalg.norm(pos0[src] - pos0[p_src]))
            continue
        zero = float(_np.linalg.norm(pos0[src] - pos0[p_src]))
        best, worst = zero, zero
        for _ in range(samples):
            q = _np.zeros(model.nq)
            m = min(model.njnt, len(lo))
            for j in range(m):
                a, b = lo[j], hi[j]
                adr = int(model.jnt_qposadr[j])
                if adr < model.nq and b > a:
                    q[adr] = rng.uniform(a, b)
            pos = fk_positions(q)
            dist = float(_np.linalg.norm(pos[src] - pos[p_src]))
            best, worst = max(best, dist), min(worst, dist)
        # A span whose joints barely change its length (a waist yaw between pelvis and hip) is rigid
        # in effect; prescribing it keeps the hips where the robot's hips are.
        if best - worst < 0.1 * best:
            rigid[src] = True
            length[src] = zero
        else:
            length[src] = best
    return parent, length, rigid


def rescale_to_robot_limbs(
    keypoints: np.ndarray,
    names: list[str],
    parent: dict[str, str],
    length: dict[str, float],
    rigid: dict[str, bool] | None = None,
) -> np.ndarray:
    """Rewrite keypoints so each segment is reachable, preserving directions.

    Rigid segments are set to the robot's exact length. Non-rigid spans are only SHORTENED when they
    exceed what the robot can reach, so the solver keeps the freedom to choose the joints in between.

    Args:
        keypoints: ``(T, K, 3)`` source keypoints, in the order of ``names``.
        names: Source-joint names matching the keypoint axis.
        parent: Source-joint -> parent source-joint (``None`` for the root).
        length: Target length (rigid) or maximum reachable distance (non-rigid).
        rigid: Source-joint -> whether its segment to the parent is rigid. All rigid when omitted.

    Returns:
        ``(T, K, 3)`` rescaled keypoints; the root keeps its world position.
    """
    idx = {n: i for i, n in enumerate(names)}
    rigid = rigid or dict.fromkeys(names, True)
    out = keypoints.copy()
    ordered, seen = [], set()

    def visit(n: str) -> None:
        if n in seen:
            return
        p = parent.get(n)
        if p is not None:
            visit(p)
        seen.add(n)
        ordered.append(n)

    for n in names:
        visit(n)

    for n in ordered:
        p = parent.get(n)
        if p is None or length.get(n, 0.0) <= 0.0:
            continue
        i, j = idx[n], idx[p]
        d = keypoints[:, i] - keypoints[:, j]
        norm = np.linalg.norm(d, axis=-1, keepdims=True)
        unit = np.divide(d, norm, out=np.zeros_like(d), where=norm > 1e-9)
        target = np.full_like(norm, length[n]) if rigid.get(n, True) else np.minimum(norm, length[n])
        out[:, i] = out[:, j] + unit * target
    return out
