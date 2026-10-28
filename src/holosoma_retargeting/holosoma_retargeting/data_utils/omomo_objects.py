"""Attach OMOMO object poses to the already-converted source clips.

The source conversion keeps only the human joints, so object-aware retargeting needs the object track
added. The object pose must land in the same frame as those joints: for OMOMO that conversion applies no
rotation and only a ground shift in z, so the object translation gets the identical shift.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import joblib
import numpy as np

from holosoma_retargeting.data_utils.amass_source import resample

# per-frame arrays of a converted clip, all cut to the object track's length
PER_FRAME_KEYS = ("global_joint_positions", "sole_normal", "sole_height")

# a cuboid's pose is ambiguous up to a 180-degree turn about any of its axes
BOX_FLIPS = np.stack([np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])])


def _angle(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Geodesic angle in rad between rotation stacks ``(..., 3, 3)``."""
    cos = (np.trace(np.swapaxes(r0, -1, -2) @ r1, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _step_limits(trans: np.ndarray, rot: np.ndarray, fps: float, max_speed: float, max_ang_speed: float) -> np.ndarray:
    """Whether each step ``t -> t+1`` breaks the speed or angular-speed limit, ``(T-1,)``."""
    speed = np.linalg.norm(np.diff(trans, axis=0), axis=1) * fps
    ang = np.degrees(_angle(rot[:-1], rot[1:])) * fps
    return (speed > max_speed) | (ang > max_ang_speed)


def _slerp_rot(r0: np.ndarray, r1: np.ndarray, w: float) -> np.ndarray:
    """Rotation a fraction ``w`` of the way from ``r0`` to ``r1``."""
    rel = r0.T @ r1
    angle = float(_angle(np.eye(3), rel))
    if angle < 1e-9:
        return r0.copy()
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2.0 * np.sin(angle))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = w * angle
    return r0 @ (np.eye(3) + np.sin(a) * k + (1.0 - np.cos(a)) * k @ k)


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian filter along axis 0, edges padded by reflection."""
    if sigma <= 0.0 or len(x) < 3:
        return x
    r = max(1, round(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], axis=1)


def clean_object_track(
    trans: np.ndarray,
    rot: np.ndarray,
    fps: float,
    max_speed: float,
    max_ang_speed: float,
    smooth_sigma: float,
    max_gap: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Remove tracking glitches from a box track: symmetry flips, dropouts, then jitter.

    Args:
        trans: ``(T, 3)`` object positions, m.
        rot: ``(T, 3, 3)`` object rotations.
        fps: Track frame rate.
        max_speed: Speed above which a step is a glitch, m/s.
        max_ang_speed: Angular speed above which a step is a glitch, deg/s.
        smooth_sigma: Gaussian smoothing width in frames; 0 disables it.
        max_gap: Longest dropout, in frames, that is interpolated.

    Returns:
        The cleaned positions and rotations, and counts of the flips, dropouts and steps still over a limit.
    """
    trans, rot = trans.copy(), rot.copy()
    report = {"flips": 0, "dropouts": 0, "residual": 0}
    # a step over the angular limit is first explained, if possible, by the box's own symmetry
    for t in range(1, len(rot)):
        if np.degrees(_angle(rot[t - 1], rot[t])) * fps <= max_ang_speed:
            continue
        cands = rot[t] @ BOX_FLIPS
        best = int(np.argmin(_angle(rot[t - 1][None], cands)))
        if best:
            rot[t:] = rot[t:] @ BOX_FLIPS[best]
            report["flips"] += 1
    # a bad step that a later bad step undoes within max_gap frames bounds a dropout: interpolate across it
    bad = _step_limits(trans, rot, fps, max_speed, max_ang_speed)
    t = 0
    while t < len(trans) - 1:
        if not bad[t]:
            t += 1
            continue
        # the run must end on the track's own jump back, not merely where a slide would stay under the limits
        for end in [e for e in range(t + 2, min(t + max_gap + 2, len(trans))) if bad[e - 1]]:
            w = (np.arange(t + 1, end) - t)[:, None] / (end - t)
            fill_t = trans[t] + w * (trans[end] - trans[t])
            fill_r = np.stack([_slerp_rot(rot[t], rot[end], float(x)) for x in w[:, 0]])
            span_t = np.concatenate([trans[t : t + 1], fill_t, trans[end : end + 1]])
            span_r = np.concatenate([rot[t : t + 1], fill_r, rot[end : end + 1]])
            if not _step_limits(span_t, span_r, fps, max_speed, max_ang_speed).any():
                trans[t + 1 : end], rot[t + 1 : end] = fill_t, fill_r
                report["dropouts"] += 1
                t = end
                break
        else:
            t += 1
    report["residual"] = int(_step_limits(trans, rot, fps, max_speed, max_ang_speed).sum())
    if smooth_sigma > 0.0:
        trans = _gaussian_smooth(trans, smooth_sigma)
        quat = rotmat_to_quat(rot)
        # one hemisphere, so the average does not cancel q against -q
        sign = np.sign(np.cumprod(np.r_[1.0, np.sign(np.sum(quat[1:] * quat[:-1], axis=1) + 1e-12)]))
        quat = _gaussian_smooth(quat * sign[:, None], smooth_sigma)
        rot = quat_to_rotmat(quat / np.linalg.norm(quat, axis=1, keepdims=True))
    return trans, rot, report


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert ``(w, x, y, z)`` quaternions ``(T, 4)`` to rotation matrices ``(T, 3, 3)``."""
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to ``(w, x, y, z)`` quaternions.

    Args:
        r: ``(T, 3, 3)`` rotation matrices.

    Returns:
        ``(T, 4)`` quaternions, w first.
    """
    t = np.trace(r, axis1=1, axis2=2)
    q = np.zeros((len(r), 4))
    big = t > 0
    s = np.sqrt(np.maximum(t[big] + 1.0, 1e-12)) * 2
    q[big, 0] = 0.25 * s
    q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / s
    q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / s
    q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / s
    for i in np.where(~big)[0]:
        m = r[i]
        k = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if k == 0:
            s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif k == 1:
            s = np.sqrt(max(1.0 - m[0, 0] + m[1, 1] - m[2, 2], 1e-12)) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(max(1.0 - m[0, 0] - m[1, 1] + m[2, 2], 1e-12)) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main() -> None:
    """Write per-clip npz carrying the human joints plus the object pose track."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--omomo", type=pathlib.Path, required=True, help="dir with *_manip_seq_joints24.p")
    ap.add_argument("--converted", type=pathlib.Path, required=True, help="existing converted OMOMO npz dir")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--objects", nargs="*", default=None, help="Only clips of these objects (default: all).")
    ap.add_argument("--max-speed", type=float, default=5.0, help="Object speed above which a step is a glitch, m/s.")
    ap.add_argument("--max-ang-speed", type=float, default=900.0, help="Angular speed of a glitch step, deg/s.")
    ap.add_argument("--smooth-sigma", type=float, default=0.5, help="Object-track smoothing, frames; 0 disables.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scales: dict[str, list[float]] = {}
    totals = {"flips": 0, "dropouts": 0, "residual": 0}
    n_ok = n_miss = n_glitch = 0
    for pkl in sorted(args.omomo.glob("*_manip_seq_joints24.p")):
        for seq in joblib.load(pkl).values():
            name = seq["seq_name"]
            if args.objects and name.split("_")[1] not in args.objects:
                continue
            src = args.converted / f"{name}.npz"
            meta_p = args.converted / f"{name}.meta.json"
            if not src.exists() or not meta_p.exists():
                n_miss += 1
                continue
            meta = json.loads(meta_p.read_text())
            if meta["rotated_to_z_up"]:
                raise ValueError(f"{name}: the joints were rotated to z-up, which this object track does not get")
            with np.load(src, allow_pickle=True) as raw:
                d = dict(raw)
            trans = np.asarray(seq["obj_trans"]).reshape(len(seq["obj_trans"]), 3).astype(np.float64)
            trans[:, 2] -= float(meta["ground"])  # same shift the joints got
            trans, rot, report = clean_object_track(
                trans,
                np.asarray(seq["obj_rot"]).astype(np.float64),
                float(meta["source_fps"]),
                args.max_speed,
                args.max_ang_speed,
                args.smooth_sigma,
            )
            for key, count in report.items():
                totals[key] += count
            if report["residual"]:
                n_glitch += 1
                continue
            quat = rotmat_to_quat(rot)
            # the same nearest-frame resample the joints got, so frame i of both is the same instant
            poses = resample(np.concatenate([quat, trans], axis=1), float(meta["source_fps"]), float(meta["fps"]))
            n = min(len(poses), len(d["global_joint_positions"]))
            for key in PER_FRAME_KEYS:
                if key in d:
                    d[key] = d[key][:n]
            d["object_poses"] = poses[:n].astype(np.float32)
            obj = name.split("_")[1]
            d["object_name"] = np.array(obj)
            d["object_scale"] = np.array(float(np.median(seq["obj_scale"])))
            np.savez(args.out / f"{name}.npz", **d)
            scales.setdefault(obj, []).append(float(np.median(seq["obj_scale"])))
            n_ok += 1
    print(f"wrote {n_ok} clips with object poses ({n_miss} source clips missing)")
    print(f"object track cleaning: {totals}; {n_glitch} clips dropped with glitches left after cleaning")
    for o, v in sorted(scales.items()):
        print(f"  {o:14s} n={len(v):4d}  scale median={np.median(v):.4f}  spread={np.min(v):.3f}-{np.max(v):.3f}")


if __name__ == "__main__":
    main()
