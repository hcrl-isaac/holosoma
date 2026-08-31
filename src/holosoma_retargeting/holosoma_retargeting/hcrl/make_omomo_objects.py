"""Generate holosoma object models for the OMOMO objects, from the shipped largebox as a template.

holosoma ships only ``largebox``; object-aware retargeting needs, per object, a urdf, the mesh, and a
combined robot+object mujoco xml. All three are name-substitutions of the largebox versions, so deriving
them keeps the structure that is known to load.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil


def main() -> None:
    """Write models/<obj>/ and models/t1/t1_23dof_w_<obj>.xml for every captured OMOMO mesh."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--captured", type=pathlib.Path, required=True, help="OMOMO captured_objects dir.")
    ap.add_argument("--models", type=pathlib.Path, required=True, help="holosoma models/ dir.")
    ap.add_argument("--scales", type=pathlib.Path, default=None, help="json of object -> mesh scale.")
    args = ap.parse_args()

    import json

    urdf_t = (args.models / "largebox" / "largebox.urdf").read_text()
    xml_t = (args.models / "t1" / "t1_23dof_w_largebox.xml").read_text()
    scales = json.loads(args.scales.read_text()) if args.scales else {}

    made = []
    for obj in sorted(args.captured.glob("*_cleaned_simplified.obj")):
        name = obj.name.replace("_cleaned_simplified.obj", "")
        if name == "largebox":
            continue
        d = args.models / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, d / f"{name}.obj")
        sc = float(scales.get(name, 1.0))
        u = urdf_t.replace("largebox", name).replace('scale="1.0 1.0 1.0"', f'scale="{sc} {sc} {sc}"')
        (d / f"{name}.urdf").write_text(u)
        x = xml_t.replace("largebox", name).replace('scale="1 1 1"', f'scale="{sc} {sc} {sc}"')
        (args.models / "t1" / f"t1_23dof_w_{name}.xml").write_text(x)
        made.append(name)
    print(f"generated {len(made)} object models: {', '.join(made)}")


if __name__ == "__main__":
    main()
