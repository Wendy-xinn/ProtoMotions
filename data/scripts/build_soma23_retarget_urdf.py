#!/usr/bin/env python3
"""Build a kinematics-only SOMA23 URDF for trajectory retargeting.

The simulation MJCF represents every non-root body with three co-located hinge
joints.  URDF only permits one joint between two links, so this script inserts
two massless intermediate links while preserving MJCF joint order and axes.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


# Conservative anatomical limits in degrees. Unlisted axes retain the MJCF
# range. SOMA uses x/y/z Euler hinges and mirrored left-side x axes.
LIMITS = {
    "RightShin_x": (0.0, 155.0),
    "RightShin_y": (-8.0, 8.0),
    "RightShin_z": (-8.0, 8.0),
    "LeftShin_x": (-155.0, 0.0),
    "LeftShin_y": (-8.0, 8.0),
    "LeftShin_z": (-8.0, 8.0),
    "RightForeArm_x": (0.0, 155.0),
    "RightForeArm_y": (-20.0, 20.0),
    "RightForeArm_z": (-35.0, 35.0),
    "LeftForeArm_x": (-155.0, 0.0),
    "LeftForeArm_y": (-20.0, 20.0),
    "LeftForeArm_z": (-35.0, 35.0),
}


def _xyz(text: str | None, default: str = "0 0 0") -> str:
    return text if text else default


def build(mjcf_path: Path, output_path: Path) -> None:
    source = ET.parse(mjcf_path).getroot()
    root_body = source.find("./worldbody/body")
    if root_body is None:
        raise ValueError(f"No root body in {mjcf_path}")

    robot = ET.Element("robot", {"name": "soma23_retarget"})
    ET.SubElement(robot, "link", {"name": "world"})

    def add_body(body: ET.Element, parent_link: str) -> None:
        body_name = body.attrib["name"]
        joints = body.findall("joint")
        if not joints:  # floating Hips root
            ET.SubElement(robot, "link", {"name": body_name})
            joint = ET.SubElement(
                robot,
                "joint",
                {"name": "world_to_Hips", "type": "fixed"},
            )
            ET.SubElement(joint, "parent", {"link": parent_link})
            ET.SubElement(joint, "child", {"link": body_name})
            ET.SubElement(joint, "origin", {"xyz": _xyz(body.get("pos")), "rpy": "0 0 0"})
        else:
            chain_parent = parent_link
            for axis_index, source_joint in enumerate(joints):
                is_last = axis_index == len(joints) - 1
                child_link = body_name if is_last else f"{body_name}__axis{axis_index}"
                ET.SubElement(robot, "link", {"name": child_link})
                name = source_joint.attrib["name"]
                joint = ET.SubElement(robot, "joint", {"name": name, "type": "revolute"})
                ET.SubElement(joint, "parent", {"link": chain_parent})
                ET.SubElement(joint, "child", {"link": child_link})
                origin = _xyz(body.get("pos")) if axis_index == 0 else "0 0 0"
                ET.SubElement(joint, "origin", {"xyz": origin, "rpy": "0 0 0"})
                ET.SubElement(joint, "axis", {"xyz": _xyz(source_joint.get("axis"), "1 0 0")})
                lo_deg, hi_deg = LIMITS.get(
                    name,
                    tuple(float(v) for v in source_joint.get("range", "-180 180").split()),
                )
                deg_to_rad = 3.141592653589793 / 180.0
                ET.SubElement(
                    joint,
                    "limit",
                    {
                        "lower": f"{lo_deg * deg_to_rad:.9f}",
                        "upper": f"{hi_deg * deg_to_rad:.9f}",
                        "effort": "1000",
                        "velocity": "20",
                    },
                )
                chain_parent = child_link

        for child in body.findall("body"):
            add_body(child, body_name)

    add_body(root_body, "world")
    ET.indent(robot, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.mjcf, args.output)


if __name__ == "__main__":
    main()
