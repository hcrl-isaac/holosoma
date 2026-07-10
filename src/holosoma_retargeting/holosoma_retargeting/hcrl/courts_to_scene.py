"""Generate holosoma climbing-task terrain files for an arena court inside each clip's seq dir.

From a fitted ``courts.json`` court (list of axis-aligned cuboid prims), writes the file set the
climbing task expects next to the clip's source npy: ``box_models/box<i>.obj`` (one mesh per prim),
``multi_boxes.obj`` (merged, interaction-mesh sampling), ``multi_boxes.urdf`` (yourdfpy collision
model), ``box_assets.xml`` + ``box_body.xml`` (mujoco includes, absolute mesh paths), and the scene
xml ``<robot>_w_multi_boxes.xml`` (base robot xml + includes + ground plane).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_XML = PKG_ROOT / "models" / "g1" / "g1_29dof_spherehand.xml"

URDF_LINK = """  <link name="multi_boxes_link_{i}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="box_models/box{i}.obj" scale="1 1 1"/></geometry>
      <material name="box{i}_material"><color rgba="0.9 0.7 0.3 0.5"/></material>
    </visual>
    <collision name="multi_boxes_{i}">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="box_models/box{i}.obj" scale="1 1 1"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="33.33"/>
      <inertia ixx="33.33" ixy="0.0" ixz="0.0" iyy="33.33" iyz="0.0" izz="33.33"/>
    </inertial>
  </link>
  <joint name="multi_boxes_joint_{i}" type="fixed">
    <parent link="world"/>
    <child link="multi_boxes_link_{i}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""


def build_court_files(court: dict, seq_dir: Path, robot_xml: Path) -> None:
    """Write the climbing terrain file set for one court into one seq dir."""
    prims = court["prims"]
    box_dir = seq_dir / "box_models"
    box_dir.mkdir(parents=True, exist_ok=True)

    meshes = []
    for i, prim in enumerate(prims, start=1):
        m = trimesh.creation.box(extents=prim["size"])
        m.apply_translation(prim["pos"])
        m.export(box_dir / f"box{i}.obj")
        meshes.append(m)
    trimesh.util.concatenate(meshes).export(seq_dir / "multi_boxes.obj")

    links = "".join(URDF_LINK.format(i=i) for i in range(1, len(prims) + 1))
    (seq_dir / "multi_boxes.urdf").write_text(
        '<?xml version="1.0"?>\n<robot name="multi_boxes">\n  <link name="world"/>\n' + links + "</robot>\n"
    )

    assets = ["<mujocoinclude>"]
    bodies = ["<mujocoinclude>"]
    for i in range(1, len(prims) + 1):
        obj_abs = (box_dir / f"box{i}.obj").resolve()
        assets.append(f'    <mesh name="box{i}" file="{obj_abs}" scale="1 1 1"/>')
        assets.append(f'    <material name="box{i}_material" rgba="0.9 0.7 0.3 0.5"/>')
        bodies.append(
            f'    <body name="multi_boxes_box{i}_link" pos="0 0 0" quat="1 0 0 0">\n'
            f'        <geom name="multi_boxes_link_{i}" type="mesh" mesh="box{i}" pos="0 0 0" quat="1 0 0 0" '
            f'material="box{i}_material" contype="1" conaffinity="1"/>\n    </body>'
        )
    assets.append("</mujocoinclude>\n")
    bodies.append("</mujocoinclude>\n")
    (seq_dir / "box_assets.xml").write_text("\n".join(assets))
    (seq_dir / "box_body.xml").write_text("\n".join(bodies))

    # scene xml: base robot xml with absolute meshdir + box includes + ground plane
    xml = robot_xml.read_text()
    meshdir_abs = (robot_xml.parent / "meshes").resolve()
    xml = xml.replace('meshdir="meshes"', f'meshdir="{meshdir_abs}"')
    assert "<asset>" in xml and "<worldbody>" in xml
    xml = xml.replace("<asset>", '<asset>\n    <include file="box_assets.xml"/>', 1)
    ground = "" if 'name="ground"' in xml else '    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0"/>\n'
    xml = xml.replace("<worldbody>", "<worldbody>\n" + ground + '    <include file="box_body.xml"/>', 1)
    scene_name = robot_xml.name.replace(".xml", "_w_multi_boxes.xml")
    (seq_dir / scene_name).write_text(xml)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--courts_json", required=True)
    ap.add_argument("--court", required=True, help="Court name whose prims become the terrain.")
    ap.add_argument("--seq_dirs", nargs="+", required=True, help="Seq dir(s) (clips on this court).")
    ap.add_argument("--robot_xml", default=str(DEFAULT_ROBOT_XML))
    args = ap.parse_args()

    court = json.loads(Path(args.courts_json).read_text())["courts"][args.court]
    for d in args.seq_dirs:
        build_court_files(court, Path(d), Path(args.robot_xml))
        print(f"[courts] {args.court} -> {d}")


if __name__ == "__main__":
    main()
