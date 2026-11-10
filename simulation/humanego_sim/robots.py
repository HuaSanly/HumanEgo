# -*- coding: utf-8 -*-
"""
Robot registry for the HumanEgo simulator.

Every robot is described by a `RobotSpec`. The simulator only relies on:
  * a MuJoCo MJCF (from MuJoCo Menagerie) whose root body is attached at the
    calibrated base pose,
  * the 6 (or 7) arm joints + their position actuators,
  * a gripper actuator with its open / closed control values,
  * a tool site at the pinch point, and
  * `R_tool_to_ee`: the rotation from that site's frame to the HumanEgo
    end-effector convention used by the real Trossen driver
    (`RobotArmTrossen.get_p`): +Z = approach, +X = finger-opening direction.

With that convention fixed, `T_align` (EE -> Aria-hand frame) is shared by every
robot, which is exactly the hardware-agnostic contract of the released
`inference/` template.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@dataclass
class RobotSpec:
    name: str
    label: str
    xml: str                       # relative to ASSETS
    root_body: str                 # body to attach at the base pose
    arm_joints: List[str]
    arm_actuators: List[str]
    gripper_actuator: str
    gripper_open: float            # actuator ctrl for fully open
    gripper_closed: float          # actuator ctrl for fully closed
    tool_site: str                 # pinch point
    R_tool_to_ee: np.ndarray       # (3,3) site frame -> HumanEgo EE frame (Z approach, X spread)
    initial_joints: Dict[str, List[float]] = field(default_factory=dict)
    home_joints: Optional[List[float]] = None
    # base mount offset applied on top of the calibrated base pose (e.g. a
    # pedestal for arms that are taller than the WidowX AI)
    mount_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gripper_force: float = 20.0    # N, clamp on the gripper actuator force
    finger_bodies: List[str] = field(default_factory=list)   # bodies that get the soft-bread adhesion proxy
    reach: float = 0.70            # m, nominal reach (for sampling / UI)
    web_name: str = ""
    # joint-space mirror (left <-> right about the base xz-plane): q_mirror = sign * q
    mirror_sign: Optional[List[float]] = None
    # gripper opening measurement: joint whose qpos maps linearly from closed -> open
    gripper_joint: str = ""
    gripper_joint_open: float = 1.0
    gripper_joint_closed: float = 0.0
    gripper_reset_qpos: Dict[str, float] = field(default_factory=dict)   # joints teleported at reset (open)
    gripper_force_param: bool = False   # whether physics.gripper_force applies to this gripper's actuator
    gripper_kp_scale: float = 1.0       # gripper servo stiffness x Menagerie (how hard it squeezes a blocked object)
    # how the controller tells "holding" from "closed on air": the gripper joint reading ("q"), or the
    # gap between the finger bodies minus their gap when closed on air ("gap"; underactuated 2F-85,
    # whose driver joint closes almost fully around a bread - the real one reports object detection)
    grip_sense: str = "q"
    finger_gap_closed: float = 0.0      # m between the finger bodies when closed on air
    max_opening: float = 0.08           # m between the finger pads when fully open (grasp skill: will the bread fit)
    kp_scale: Optional[float] = None    # None -> physics.arm_kp_scale from the config
    attach: List[dict] = field(default_factory=list)   # sub-models attached to a site, e.g. a gripper
    tool_site_spec: Optional[dict] = None   # {"body":..., "pos":[...]} adds the tool site if the MJCF has none
    fetch: Optional[dict] = None        # where to download the assets from (scripts/fetch_robots.py)
    supported_modes: tuple = ("single", "bimanual")
    setup_command: str = ""            # local model import, when not supplied by Menagerie


# Real Trossen driver: T_new = T_raw @ C  with  X_new=Y_raw, Y_new=Z_raw, Z_new=X_raw
# (see AriaMimic inference/RobotArmTrossen._get_correction_matrix). The MuJoCo
# `ee_gripper_link` site has link_6's orientation, i.e. the driver's raw frame.
_TROSSEN_C = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

# Franka hand / Robotiq 2F-85: approach = site +Z, fingers open along site +Y
#   -> EE X = site Y, EE Y = -site X, EE Z = site Z
_SPREAD_Y = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])

ROBOTS: Dict[str, RobotSpec] = {
    "brx20260825": RobotSpec(
        name="brx20260825", label="BRX 20260825 (right arm)",
        xml="robots/brx20260825/brx.xml", root_body="base_link",
        arm_joints=[f"ArmR{i:02d}_Joint" for i in range(2, 9)],
        arm_actuators=[f"ArmR{i:02d}_Joint" for i in range(2, 9)],
        gripper_actuator="gripper", gripper_open=0.0, gripper_closed=0.041,
        tool_site="tcp", R_tool_to_ee=np.eye(3),
        initial_joints={"right": [-0.40856, -0.96783, -0.36714, 0.23111, 0.30946, -0.39722, 0.23402]},
        home_joints=[-0.40856, -0.96783, -0.36714, 0.23111, 0.30946, -0.39722, 0.23402],
        finger_bodies=["JawBlock01_Link", "JawBlock02_Link"],
        gripper_joint="JawBlock01_Joint", gripper_joint_open=0.0, gripper_joint_closed=0.041,
        gripper_reset_qpos={"JawBlock01_Joint": 0.0, "JawBlock02_Joint": 0.0},
        gripper_force=35.0, gripper_force_param=True, max_opening=0.082,
        kp_scale=1.0, reach=0.70, supported_modes=("single",),
        setup_command="python simulation/scripts/import_brx.py --source ~/BRX20260825",
    ),
    "trossen_wxai": RobotSpec(
        name="trossen_wxai",
        label="Trossen WidowX AI",
        xml="robots/trossen_wxai/wxai_follower.xml",
        root_body="base_link",
        arm_joints=[f"joint_{i}" for i in range(6)],
        arm_actuators=[f"joint_{i}" for i in range(6)],
        gripper_actuator="right_carriage_joint",
        gripper_open=0.044,
        gripper_closed=-0.002,
        tool_site="ee_gripper_link",
        R_tool_to_ee=_TROSSEN_C,
        # cfg/inference/serve_bread/RobotArmTrossen{Left,Right}.yaml (real rig)
        initial_joints={
            "right": [-0.622, 2.511, 1.697, -1.526, -0.936, -2.526],
            "left": [0.470, 2.340, 1.589, -1.283, 0.812, 2.300],   # rig.mirror_left_start: mirror of "right"
        },
        home_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_force=12.0,
        max_opening=0.088,
        reach=0.70,
        finger_bodies=["gripper_left", "gripper_right"],
        web_name="trossen_wxai",
        gripper_joint="right_carriage_joint",
        gripper_joint_open=0.044,
        gripper_joint_closed=0.0,
        gripper_reset_qpos={"right_carriage_joint": 0.044, "left_carriage_joint": 0.044},
        gripper_force_param=True,
        mirror_sign=[-1, 1, 1, 1, -1, -1],
    ),
    # Franka Emika Panda + Franka Hand (MuJoCo Menagerie). Fingers slide along the hand's y axis.
    "franka_panda": RobotSpec(
        name="franka_panda",
        label="Franka Panda",
        xml="robots/franka_panda/panda.xml",
        root_body="link0",
        arm_joints=[f"joint{i}" for i in range(1, 8)],
        arm_actuators=[f"actuator{i}" for i in range(1, 8)],
        gripper_actuator="actuator8",
        gripper_open=255.0,
        gripper_closed=0.0,
        tool_site="tcp",
        tool_site_spec={"body": "hand", "pos": [0.0, 0.0, 0.1085]},
        R_tool_to_ee=_SPREAD_Y,
        initial_joints={},           # filled by IK from the Trossen start pose (see robot_poses.json)
        home_joints=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        # Menagerie's hand servo (100 N/m on the finger tendon) squeezes a 6 cm bread with only ~3 N
        # (the real Franka Hand: 70 N continuous). A stiffer servo (physics.robot_gripper.franka_panda
        # .kp_scale 2-25) did not raise soft-bread success, so the Menagerie servo is kept.
        gripper_force=40.0,
        max_opening=0.08,
        reach=0.85,
        finger_bodies=["left_finger", "right_finger"],
        gripper_joint="finger_joint1",
        gripper_joint_open=0.04,
        gripper_joint_closed=0.0,
        gripper_reset_qpos={"finger_joint1": 0.04, "finger_joint2": 0.04},
        kp_scale=1.0,
        fetch={"menagerie": "franka_emika_panda", "files": ["panda.xml", "assets", "LICENSE"]},
    ),
    # Universal Robots UR5e + Robotiq 2F-85 (MuJoCo Menagerie), gripper on the flange site.
    "ur5e": RobotSpec(
        name="ur5e",
        label="UR5e + Robotiq 2F-85",
        xml="robots/ur5e/ur5e.xml",
        root_body="base",
        arm_joints=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        arm_actuators=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        gripper_actuator="gripper/fingers_actuator",
        gripper_open=0.0,
        gripper_closed=255.0,
        tool_site="gripper/pinch",
        R_tool_to_ee=_SPREAD_Y,
        initial_joints={},
        home_joints=[0.0, -1.9, 1.9, -1.571, -1.571, 0.0],
        gripper_force=40.0,          # Menagerie caps the 2F-85 tendon at +-5; the real gripper squeezes far harder
        max_opening=0.085,
        reach=0.85,
        finger_bodies=["gripper/left_pad", "gripper/right_pad"],
        grip_sense="gap",
        finger_gap_closed=0.0149,
        gripper_force_param=True,
        gripper_joint="gripper/right_driver_joint",
        gripper_joint_open=0.0,
        gripper_joint_closed=0.8,
        kp_scale=1.0,
        attach=[{"xml": "robots/robotiq_2f85/2f85.xml", "site": "attachment_site", "prefix": "gripper/"}],
        fetch={"menagerie": "universal_robots_ur5e", "files": ["ur5e.xml", "assets", "LICENSE"],
               "extra": [{"menagerie": "robotiq_2f85", "files": ["2f85.xml", "assets", "LICENSE"],
                          "dest": "robotiq_2f85"}]},
    ),
}


def get_robot(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise KeyError(f"unknown robot '{name}'. Available: {list(ROBOTS)}")
    r = ROBOTS[name]
    if not os.path.exists(os.path.join(ASSETS, r.xml)):
        command = r.setup_command or f"python simulation/scripts/fetch_robots.py {name}"
        raise FileNotFoundError(f"{r.label} assets missing: run `{command}`")
    # start poses for robots without real-rig values: IK-matched to the Trossen start pose
    if not r.initial_joints:
        import json
        p = os.path.join(ASSETS, "robots", "robot_poses.json")
        if os.path.exists(p):
            with open(p) as f:
                poses = json.load(f)
            if name in poses:
                r.initial_joints = poses[name]["initial"]
    return r
