"""Convert holosoma retargeting output (qpos npz, 30 fps) back to BONES-SEED csv layout (120 fps).

qpos rows are mujoco freejoint (xyz + wxyz quat) + 29 hinge joints in MODEL order; the csv wants
(px py pz qx qy qz qw + joints in the csv column order) at the source rate. Root position and joints
are cubic-interpolated, the quaternion slerped, from 30 fps back up to 120 fps.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from holosoma_retargeting.hcrl.csv_to_g1fk import DEFAULT_MODEL


def qpos_to_csv(qpos: np.ndarray, model: mujoco.MjModel, columns: list[str], up: int) -> pd.DataFrame:
    """Upsample (T,36) qpos by ``up`` and reorder into the csv column layout."""
    t = np.arange(len(qpos))
    t_hi = np.linspace(0, len(qpos) - 1, (len(qpos) - 1) * up + 1)

    pos = PchipInterpolator(t, qpos[:, :3])(t_hi)  # monotone: no overshoot at direction changes
    rot = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])  # wxyz -> xyzw for scipy
    quat = Slerp(t, rot)(t_hi).as_quat()  # xyzw
    joints_model = PchipInterpolator(t, qpos[:, 7:])(t_hi)

    # model hinge order -> csv column order
    model_joints = [model.joint(j).name for j in range(model.njnt) if model.joint(j).type != mujoco.mjtJoint.mjJNT_FREE]
    joint_cols = columns[7:]
    idx = [model_joints.index(c) for c in joint_cols]
    return pd.DataFrame(np.concatenate([pos, quat, joints_model[:, idx]], axis=1), columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qpos_npz", nargs="+", required=True, help="Retargeting output npz file(s).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--columns_from", required=True, help="A source csv whose column layout to reproduce.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--upsample", type=int, default=4, help="30 fps solve -> 120 fps csv.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    columns = list(pd.read_csv(args.columns_from, nrows=0).columns)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.qpos_npz:
        d = np.load(f, allow_pickle=True)
        df = qpos_to_csv(np.asarray(d["qpos"]), model, columns, args.upsample)
        stem = Path(f).stem.replace("_original", "")
        df.to_csv(out_dir / f"{stem}.csv", index=False)
        print(f"[qpos2csv] {stem}: {d['qpos'].shape} -> {len(df)} rows")


if __name__ == "__main__":
    main()
