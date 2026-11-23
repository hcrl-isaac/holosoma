"""Write holosoma's T2 retarget model (MJCF + URDF) from booster_assets ``robots/T2/T2_31dof``.

Adds the sole spheres and palm points the solver keys on, unnames the free joint (joint limits are read from named
joints only) and names the floor ``ground`` (the ground constraints find it by name).

    python -m holosoma_retargeting.data_utils.make_t2_model <booster_assets>/robots/T2/T2_31dof models/t2
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

R = 0.005
SOLE_Z = -0.0325 + R  # foot box bottom in *_ankle_roll_link
FOOT = {1: (-0.055, 0.03), 2: (-0.055, -0.03), 3: (0.135, 0.03), 4: (0.135, -0.03), 5: (0.145, 0.0)}
PALM_Y = 0.08


def spheres(side):
    out = [(f"{side}_foot_sphere_{i}_link", f"{side}_ankle_roll_link", (x, y, SOLE_Z)) for i, (x, y) in FOOT.items()]
    s = 1 if side == "left" else -1
    out.append((f"{side}_hand_sphere_link", f"{side}_wrist_roll_link", (0.0, s * PALM_Y, -0.01)))
    return out


EXTRA = spheres("left") + spheres("right")


def fmt(p: tuple) -> str:
    return " ".join(f"{v:g}" for v in p)


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print ``elem`` in place (``ET.indent`` is not in the 3.8 stdlib mypy checks against)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def main(src: str, dst: str) -> None:
    # MJCF
    t = ET.parse(f"{src}/T2_31dof.xml")  # noqa: S314 (booster_assets, a pinned local checkout)
    r = t.getroot()
    r.set("model", "t2_31dof")
    bodies = {b.get("name"): b for b in r.iter("body")}
    # holosoma reads joint limits from named joints only, so the free joint must be unnamed (as in its T1 model)
    for j in r.iter("joint"):
        if j.get("type") == "free":
            j.attrib.pop("name", None)
    # the solver's ground constraints find the floor by a geom name containing "ground"
    for g in r.iter("geom"):
        if g.get("name") == "floor":
            g.set("name", "ground")
    for name, parent, pos in EXTRA:
        b = ET.SubElement(bodies[parent], "body", name=name, pos=fmt(pos))
        ET.SubElement(b, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-7 1e-7 1e-7")
        ET.SubElement(
            b, "geom", name=name, type="sphere", size=f"{R}", rgba="0.2 0.2 0.2 1", contype="1", conaffinity="1"
        )
    _indent(t.getroot())
    t.write(f"{dst}/t2_31dof.xml")
    with open(f"{dst}/t2_31dof.xml", "a") as f:
        f.write("\n")

    # URDF
    t = ET.parse(f"{src}/T2_31dof.urdf")  # noqa: S314
    r = t.getroot()
    r.set("name", "t2_31dof")
    for name, parent, pos in EXTRA:
        link = ET.SubElement(r, "link", name=name)
        i = ET.SubElement(link, "inertial")
        ET.SubElement(i, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(i, "mass", value="0.001")
        ET.SubElement(i, "inertia", ixx="1e-7", ixy="0", ixz="0", iyy="1e-7", iyz="0", izz="1e-7")
        c = ET.SubElement(link, "collision")
        ET.SubElement(c, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(c, "geometry"), "sphere", radius=f"{R}")
        j = ET.SubElement(r, "joint", name=name.replace("_link", "_joint"), type="fixed")
        ET.SubElement(j, "origin", xyz=fmt(pos), rpy="0 0 0")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=name)
    _indent(t.getroot())
    t.write(f"{dst}/t2_31dof.urdf")
    with open(f"{dst}/t2_31dof.urdf", "a") as f:
        f.write("\n")
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
