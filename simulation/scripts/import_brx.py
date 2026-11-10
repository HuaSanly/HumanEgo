"""Prepare the local BRX CAD export for the single-right-arm MuJoCo scene.

    python simulation/scripts/import_brx.py --source /home/huasan/BRX20260825

The source export is never modified. Meshes and the generated model are local
build assets, like the optional Menagerie robots. Motor gains/forces below are
simulation settings, not hardware specifications (the URDF supplies zeros).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from humanego_sim.robots import ASSETS  # noqa: E402

ARM_JOINTS = [f"ArmR{i:02d}_Joint" for i in range(2, 9)]
JAW_JOINTS = ["JawBlock01_Joint", "JawBlock02_Joint"]
# Lean the fixed torso towards the table; park the inactive arm at its CAD pose.
# These joints are removed, not left passive. Values are radians.
PARKED_JOINTS = {"FoldingModularJoint02_Joint": -0.35,
                 "FoldingModularJoint03_Joint": -0.19661, "Trunk_Joint": -0.36182}


def import_brx(source: Path, output: Path) -> Path:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    urdf = source / "urdf" / "BRX20260825.urdf"
    root = ET.parse(urdf).getroot()
    expected = set(ARM_JOINTS + JAW_JOINTS)
    if not expected <= {j.get("name") for j in root.findall("joint")}:
        raise ValueError("BRX export is missing right-arm or jaw joints")
    # Validate all resources before writing the destination.
    meshes = {}
    for mesh in root.findall(".//mesh"):
        name = Path(mesh.get("filename")).name
        path = source / "meshes" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        meshes[name] = path
        mesh.set("filename", name)
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed" and joint.get("name") not in expected:
            angle = PARKED_JOINTS.get(joint.get("name"), 0.0)
            if angle:
                origin = joint.find("origin")
                rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                rotation = rotation * Rotation.from_rotvec(axis * angle)
                origin.set("rpy", " ".join(map(str, rotation.as_euler("xyz"))))
            joint.set("type", "fixed")
    # Visual CAD meshes remain intact. Use small distal finger boxes for contact,
    # avoiding a convex hull across each jaw's mounting/rail geometry.
    for index, name in enumerate(["JawBlock01_Link", "JawBlock02_Link"]):
        collision = root.find(f"link[@name='{name}']/collision")
        collision.clear()
        ET.SubElement(collision, "origin", xyz=f"0.073 0 {0.004 if index == 0 else -0.004}", rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.02 0.024 0.008")
    output.mkdir(parents=True, exist_ok=True)
    mesh_dir = output / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    for name, path in meshes.items():
        shutil.copyfile(path, mesh_dir / name)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", meshdir=str(mesh_dir), strippath="true",
                  discardvisual="false", fusestatic="false")
    with tempfile.TemporaryDirectory(prefix="brx_import_") as tmp:
        path = Path(tmp) / "brx.urdf"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(path))
        spec = mujoco.MjSpec.from_file(str(path))
    data = mujoco.MjData(model)
    for name, value in zip(JAW_JOINTS, [0.041, -0.041]):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    # TCP = centre of the opposed pads at their distal gripping area.
    parent = model.body("LinearclampinggripperJZ01_Link").id
    jaw = model.body("JawBlock01_Link").id
    jaw_rotation = data.xmat[jaw].reshape(3, 3)
    centres = [data.xpos[model.body(n).id] for n in ("JawBlock01_Link", "JawBlock02_Link")]
    tcp_world = np.mean(centres, axis=0) + jaw_rotation @ np.array([0.073, 0, 0])
    parent_rotation = data.xmat[parent].reshape(3, 3)
    site = spec.body("LinearclampinggripperJZ01_Link").add_site()
    site.name = "tcp"
    site.pos = parent_rotation.T @ (tcp_world - data.xpos[parent])
    # HumanEgo: X spreads the fingers; Z approaches along the fingers' length.
    tool_rotation = parent_rotation.T @ jaw_rotation @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    xyzw = Rotation.from_matrix(tool_rotation).as_quat()
    site.quat = np.r_[xyzw[3], xyzw[:3]]
    site.size = [0.004, 0, 0]
    site.group = 5
    # The trunk is an open frame and the wrist camera includes a bracket. Keep
    # separate CAD components separate instead of filling their gaps with one
    # convex collision hull. Original meshes still provide the visuals.
    for name in ("Trunk_Link", "ArmRcamera_link", "ArmLcamera_link"):
        body = spec.body(name)
        for geom in list(body.geoms):
            if geom.contype or geom.conaffinity:
                spec.delete(geom)
        mesh = trimesh.load(mesh_dir / f"{name}.STL", force="mesh", process=True)
        for index, part in enumerate(mesh.split(only_watertight=False)):
            collision_name = f"{name}_collision_{index}"
            file_name = collision_name + ".obj"
            part.convex_hull.export(mesh_dir / file_name)
            spec.add_mesh(name=collision_name, file=str(mesh_dir / file_name))
            body.add_geom(name=collision_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=collision_name)
    # CAD wrist housings are concave. Their single convex hull overlaps other
    # wrist parts at valid joint angles; use link-axis capsules for contact.
    for side in ("R", "L"):
        for number in range(2, 9):
            body = spec.body(f"Arm{side}{number:02d}_Link")
            for geom in list(body.geoms):
                if geom.contype or geom.conaffinity:
                    spec.delete(geom)
            child_joint = (f"Arm{side}{number + 1:02d}_Joint" if number < 8
                           else f"LinearclampinggripperJZ0{1 if side == 'R' else 2}_Joint")
            origin = root.find(f"joint[@name='{child_joint}']/origin")
            end = np.fromstring(origin.get("xyz"), sep=" ")
            body.add_geom(name=f"Arm{side}{number:02d}_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=np.r_[np.zeros(3), end], size=[0.025 if number < 7 else 0.014, 0, 0])
    for joint in spec.joints:
        joint.damping[0] = 2.0 if joint.name in ARM_JOINTS else 1.0
        joint.armature = 0.02 if joint.name in ARM_JOINTS else 0.001
    for name, kp, kv, force in zip(ARM_JOINTS, [600, 600, 500, 500, 300, 200, 150],
                                  [45, 45, 35, 35, 20, 15, 10], [80, 80, 60, 60, 30, 20, 20]):
        actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.set_to_position(kp=kp, kv=kv)
        actuator.ctrlrange = model.jnt_range[model.joint(name).id]
        actuator.ctrllimited = True
        actuator.forcerange = [-force, force]
        actuator.forcelimited = True
    actuator = spec.add_actuator(name="gripper", target=JAW_JOINTS[0], trntype=mujoco.mjtTrn.mjTRN_JOINT)
    actuator.set_to_position(kp=2500, kv=20)
    actuator.ctrlrange = [0, 0.041]
    actuator.ctrllimited = True
    actuator.forcerange = [-35, 35]
    actuator.forcelimited = True
    coupling = spec.add_equality(name="jaw_coupling", type=mujoco.mjtEq.mjEQ_JOINT,
                                 name1=JAW_JOINTS[1], name2=JAW_JOINTS[0])
    coupling.data[:5] = [0, -1, 0, 0, 0]
    coupling.solref = [0.004, 1]
    # CAD parts touch at assembly seams. Exclude adjacent bodies and the short
    # wrist assemblies, retaining arm/table, finger/object and other self-contact.
    for body in spec.bodies:
        if body.name != "world" and body.parent.name != "world":
            spec.add_exclude(bodyname1=body.name, bodyname2=body.parent.name)
    for side in ("R", "L"):
        spec.add_exclude(bodyname1=f"Arm{side}06_Link", bodyname2=f"Arm{side}08_Link")
    spec.meshdir = "meshes"
    spec.modelfiledir = str(output)
    for mesh in spec.meshes:
        mesh.file = Path(mesh.file).name
    spec.compile()
    destination = output / "brx.xml"
    destination.write_text(spec.to_xml(), encoding="utf-8")
    provenance = {"source_urdf": urdf.name, "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                  "mujoco_version": mujoco.__version__, "active_joints": ARM_JOINTS + JAW_JOINTS,
                  "parked_joints": PARKED_JOINTS, "simulation_parameters_only": True,
                  "mesh_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in sorted(meshes.items())}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / "BRX20260825")
    parser.add_argument("--output", type=Path, default=Path(ASSETS) / "robots" / "brx20260825")
    args = parser.parse_args()
    print(import_brx(args.source, args.output))
