"""Attach OMOMO object poses to the already-converted source clips.

The source conversion kept only the human joints, so object-aware retargeting had nothing to hold onto.
The object pose must land in the SAME frame as those joints: for OMOMO that conversion applied no
rotation and only a ground shift in z, so the object translation gets the identical shift.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import joblib
import numpy as np


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


def aa_to_quat(v: np.ndarray) -> np.ndarray:
    """Axis-angle to ``(w, x, y, z)`` quaternions.

    Args:
        v: ``(T, 3)`` axis-angle rotations.

    Returns:
        ``(T, 4)`` unit quaternions, w first.
    """
    th = np.linalg.norm(v, axis=1, keepdims=True)
    axis = np.divide(v, th, out=np.zeros_like(v), where=th > 1e-9)
    half = th / 2.0
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=1)


def main() -> None:
    """Write per-clip npz carrying the human joints plus the object pose track."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--omomo", type=pathlib.Path, required=True, help="dir with *_manip_seq_joints24.p")
    ap.add_argument("--converted", type=pathlib.Path, required=True, help="existing converted OMOMO npz dir")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scales: dict[str, list[float]] = {}
    n_ok = n_miss = 0
    for pkl in sorted(args.omomo.glob("*_manip_seq_joints24.p")):
        for seq in joblib.load(pkl).values():
            name = seq["seq_name"]
            src = args.converted / f"{name}.npz"
            meta_p = args.converted / f"{name}.meta.json"
            if not src.exists() or not meta_p.exists():
                n_miss += 1
                continue
            d = dict(np.load(src, allow_pickle=True))
            ground = float(json.loads(meta_p.read_text())["ground"])
            trans = np.asarray(seq["obj_trans"]).reshape(len(seq["obj_trans"]), 3).astype(np.float64)
            trans[:, 2] -= ground  # same shift the joints got
            quat = rotmat_to_quat(np.asarray(seq["obj_rot"]).astype(np.float64))
            n = min(len(trans), len(d["global_joint_positions"]))
            d["object_poses"] = np.concatenate([quat[:n], trans[:n]], axis=1).astype(np.float32)
            # the source root orientation, which the joints-only conversion had dropped
            d["root_quat"] = aa_to_quat(np.asarray(seq["root_orient"]).astype(np.float64))[:n].astype(np.float32)
            obj = name.split("_")[1]
            d["object_name"] = np.array(obj)
            d["object_scale"] = np.array(float(np.median(seq["obj_scale"])))
            np.savez(args.out / f"{name}.npz", **d)
            scales.setdefault(obj, []).append(float(np.median(seq["obj_scale"])))
            n_ok += 1
    print(f"wrote {n_ok} clips with object poses ({n_miss} source clips missing)")
    for o, v in sorted(scales.items()):
        print(f"  {o:14s} n={len(v):4d}  scale median={np.median(v):.4f}  spread={np.min(v):.3f}-{np.max(v):.3f}")


if __name__ == "__main__":
    main()
