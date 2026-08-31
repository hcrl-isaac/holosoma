"""SMPL forward kinematics for pose-parameter motion sources (no chumpy, no body-model deps).

Joint positions depend only on the shaped rest joints and the kinematic tree, so the shape/pose
blend shapes and skinning needed for vertices are skipped.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np

# SMPL body joints 0-21 in model order; this is exactly holosoma's SMPLX_DEMO_JOINTS.
SMPL_BODY_JOINTS = 22

# SMPL rest frame is right-handed with +X = subject left, +Y = up, +Z = forward.
# Cyclic permutation to the robot convention (+X forward, +Y left, +Z up) preserves handedness,
# unlike utils.transform_y_up_to_z_up which mirrors (it targets left-handed sources).
_Y_UP_TO_Z_UP = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _install_chumpy_shim() -> None:
    """Register a minimal ``chumpy.ch.Ch`` so the official SMPL pickles load on modern numpy."""
    if "chumpy.ch" in sys.modules:
        return

    class Ch:
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

        def __array__(self, dtype=None):
            return np.asarray(self.__dict__.get("x"), dtype=dtype)

    pkg, mod = types.ModuleType("chumpy"), types.ModuleType("chumpy.ch")
    mod.Ch = Ch
    pkg.ch = mod
    sys.modules.setdefault("chumpy", pkg)
    sys.modules["chumpy.ch"] = mod


def load_smpl_model(model_path: str | Path) -> dict:
    """Load rest joints, parents and subject height from an SMPL ``.pkl``.

    Args:
        model_path: Path to a SMPL model pickle (e.g. ``SMPL_MALE.pkl``).

    Returns:
        Dict with ``J_rest`` (24, 3), ``parents`` (24,), ``height`` (metres), ``v_template`` and
        ``weights`` (the last two for skinning foot vertices).
    """
    _install_chumpy_shim()
    with open(model_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    v_template = np.asarray(data["v_template"], dtype=np.float64)
    regressor = data["J_regressor"]
    regressor = np.asarray(regressor.todense()) if hasattr(regressor, "todense") else np.asarray(regressor)
    parents = np.asarray(data["kintree_table"], dtype=np.int64)[0].copy()
    parents[0] = -1  # root sentinel is stored as uint32 max
    return {
        "J_rest": regressor @ v_template,
        "parents": parents,
        "height": float(v_template[:, 1].max() - v_template[:, 1].min()),
        "v_template": v_template,
        "weights": np.asarray(data["weights"], dtype=np.float64),
    }


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to rotation matrices.

    Args:
        rotvec: Array of shape (..., 3).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(rotvec, theta, out=np.zeros_like(rotvec), where=theta > 0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(*axis.shape[:-1], 3, 3)
    eye = np.broadcast_to(np.eye(3), skew.shape)
    sin, cos = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def smpl_joint_positions(model: dict, poses: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Run SMPL forward kinematics to global joint positions in the SMPL frame.

    ``trans`` is taken as the pelvis position, so the rest-pose root offset is removed first.

    Args:
        model: Output of :func:`load_smpl_model`.
        poses: Axis-angle pose parameters, shape (T, 72).
        trans: Root translation, shape (T, 3).

    Returns:
        Global joint positions of shape (T, 24, 3).
    """
    n_joints = model["J_rest"].shape[0]
    rot = _rodrigues(np.asarray(poses, dtype=np.float64).reshape(-1, n_joints, 3))
    j_rest, parents = model["J_rest"], model["parents"]

    global_rot = [rot[:, 0]]
    global_pos = [np.broadcast_to(j_rest[0], rot.shape[:1] + (3,)).copy()]
    for i in range(1, n_joints):
        p = parents[i]
        global_rot.append(global_rot[p] @ rot[:, i])
        global_pos.append(global_pos[p] + np.einsum("tij,j->ti", global_rot[p], j_rest[i] - j_rest[p]))
    joints = np.stack(global_pos, axis=1) - j_rest[0]
    return joints + np.asarray(trans, dtype=np.float64)[:, None, :]


def to_z_up(points: np.ndarray) -> np.ndarray:
    """Rotate SMPL-frame points into the robot z-up frame, preserving handedness.

    Args:
        points: Array of shape (..., 3).

    Returns:
        Rotated array of the same shape.
    """
    return points @ _Y_UP_TO_Z_UP.T


# SMPL joints whose skinning defines each foot, and the sole fraction of those vertices.
_FOOT_JOINTS = {"left": (7, 10), "right": (8, 11)}
_SOLE_QUANTILE = 25


def foot_vertices(model: dict) -> dict[str, np.ndarray]:
    """Vertex indices skinned to each foot, i.e. the surface a ball can touch.

    Args:
        model: Output of :func:`load_smpl_model`.

    Returns:
        Per-side arrays of vertex indices.
    """
    weights = model["weights"]
    return {side: np.flatnonzero(sum(weights[:, j] for j in joints) > 0.5) for side, joints in _FOOT_JOINTS.items()}


def sole_vertices(model: dict) -> dict[str, np.ndarray]:
    """Vertex indices forming each foot's sole, taken as the lowest band of its skinned vertices.

    Args:
        model: Output of :func:`load_smpl_model`.

    Returns:
        Per-side arrays of vertex indices.
    """
    v_template = model["v_template"]
    out = {}
    for side, idx in foot_vertices(model).items():
        # the rest pose is y-up, so the sole is the low-y band
        out[side] = idx[v_template[idx, 1] < np.percentile(v_template[idx, 1], _SOLE_QUANTILE)]
    return out


def skin_vertices(model: dict, poses: np.ndarray, trans: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    """Skin a subset of SMPL vertices, so foot geometry is available without posing the whole mesh.

    Args:
        model: Output of :func:`load_smpl_model`.
        poses: Axis-angle pose parameters, shape (T, 72).
        trans: Root translation, shape (T, 3).
        vertex_ids: Vertex indices to skin.

    Returns:
        Positions of shape (T, len(vertex_ids), 3) in the SMPL frame.
    """
    j_rest, parents = model["J_rest"], model["parents"]
    n_joints = j_rest.shape[0]
    rot = _rodrigues(np.asarray(poses, dtype=np.float64).reshape(-1, n_joints, 3))

    global_rot, global_pos = [rot[:, 0]], [np.broadcast_to(j_rest[0], rot.shape[:1] + (3,)).copy()]
    for i in range(1, n_joints):
        p = parents[i]
        global_rot.append(global_rot[p] @ rot[:, i])
        global_pos.append(global_pos[p] + np.einsum("tij,j->ti", global_rot[p], j_rest[i] - j_rest[p]))
    global_rot, global_pos = np.stack(global_rot, 1), np.stack(global_pos, 1)
    offset = global_pos - np.einsum("tkij,kj->tki", global_rot, j_rest)  # rest-pose correction

    w = model["weights"][vertex_ids]
    verts = model["v_template"][vertex_ids]
    rot_v = np.einsum("vk,tkij->tvij", w, global_rot)
    off_v = np.einsum("vk,tki->tvi", w, offset)
    return np.einsum("tvij,vj->tvi", rot_v, verts) + off_v - j_rest[0] + np.asarray(trans, np.float64)[:, None, :]


def plane_normals(points: np.ndarray) -> np.ndarray:
    """Unit normal of the best-fit plane through each frame's point set, oriented upward.

    Args:
        points: Array of shape (T, N, 3), already in a z-up frame.

    Returns:
        Normals of shape (T, 3).
    """
    centred = points - points.mean(axis=1, keepdims=True)
    normals = np.linalg.svd(centred, full_matrices=False)[2][:, 2, :]
    return normals * np.sign(normals[:, 2])[:, None]
