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

from holosoma_retargeting.hcrl.amass_source import _resample

# per-frame arrays of a converted clip, all cut to the object track's length
PER_FRAME_KEYS = ("global_joint_positions", "sole_normal", "sole_height")


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
            meta = json.loads(meta_p.read_text())
            if meta["rotated_to_z_up"]:
                raise ValueError(f"{name}: the joints were rotated to z-up, which this object track does not get")
            with np.load(src, allow_pickle=True) as raw:
                d = dict(raw)
            trans = np.asarray(seq["obj_trans"]).reshape(len(seq["obj_trans"]), 3).astype(np.float64)
            trans[:, 2] -= float(meta["ground"])  # same shift the joints got
            quat = rotmat_to_quat(np.asarray(seq["obj_rot"]).astype(np.float64))
            # the same nearest-frame resample the joints got, so frame i of both is the same instant
            poses = _resample(np.concatenate([quat, trans], axis=1), float(meta["source_fps"]), float(meta["fps"]))
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
    for o, v in sorted(scales.items()):
        print(f"  {o:14s} n={len(v):4d}  scale median={np.median(v):.4f}  spread={np.min(v):.3f}-{np.max(v):.3f}")


if __name__ == "__main__":
    main()
