"""LAFAN1 world joint positions -> holosoma's ``lafan`` source files, without the mirror image.

holosoma's lafan loader turns y-up into z-up by swapping y and z, a reflection, and its keypoint list names
the joints right-for-left to match, so every take comes out mirrored: the robot gestures with the other
hand. Written here in ``LAFAN_DEMO_JOINTS`` order by name with the forward axis negated, the loader's
transform lands on a proper rotation instead.

    python -m holosoma_retargeting.hcrl.lafan_source --bvh-dir <bvh> --positions-dir <npy> --out-dir <src> \
        sprint1_subject2 walk3_subject4:198:212
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS

FPS = 30.0
ALIASES = {"LeftToe": "LeftToeBase", "RightToe": "RightToeBase"}


def bvh_joint_names(bvh_path: Path) -> list[str]:
    """Joint names in the order the BVH declares them (its positions follow the same order)."""
    names = []
    for line in bvh_path.read_text().splitlines():
        words = line.split()
        if words and words[0] in ("ROOT", "JOINT"):
            names.append(ALIASES.get(words[1], words[1]))
    return names


def to_holosoma(positions: np.ndarray, names: list[str], start_s: float | None = None,
                end_s: float | None = None) -> np.ndarray:
    """Reorder, un-mirror and crop one take.

    Args:
        positions: ``(T, J, 3)`` y-up world positions in meters, in ``names`` order.
        names: The BVH joint names.
        start_s: Window start in seconds, or None for the take's start.
        end_s: Window end in seconds, or None for the take's end.

    Returns:
        ``(T', len(LAFAN_DEMO_JOINTS), 3)`` positions for holosoma's lafan loader.
    """
    missing = [n for n in LAFAN_DEMO_JOINTS if n not in names]
    if missing:
        raise ValueError(f"BVH lacks joints {missing}")
    out = positions[:, [names.index(n) for n in LAFAN_DEMO_JOINTS]].copy()
    out[..., 2] *= -1.0
    a = 0 if start_s is None else round(start_s * FPS)
    b = len(out) if end_s is None else round(end_s * FPS)
    return out[a:b]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--positions-dir", type=Path, required=True, help="extract_global_positions.py output")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("takes", nargs="+", help="take, or take:start_s:end_s for a window")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.takes:
        take, *window = spec.split(":")
        start_s, end_s = (float(window[0]), float(window[1])) if window else (None, None)
        names = bvh_joint_names(args.bvh_dir / f"{take}.bvh")
        positions = np.load(args.positions_dir / f"{take}.npy")
        stem = take if not window else f"{take}_{window[0]}-{window[1]}s"
        out = to_holosoma(positions, names, start_s, end_s)
        np.save(args.out_dir / f"{stem}.npy", out)
        print(f"wrote {stem}.npy: {out.shape[0]} frames ({out.shape[0] / FPS:.1f} s)")


if __name__ == "__main__":
    main()
