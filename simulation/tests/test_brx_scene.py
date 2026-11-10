"""BRX frame calibration, fixed-body assembly and actual right-arm physics."""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanego_sim.cameras import get_camera
from humanego_sim.driver import T_to_p
from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.robots import ASSETS, ROBOTS
from humanego_sim.scene import build_scene, rig_geometry

CONFIG = str(Path(SIM_ROOT) / "configs" / "pick_cube_brx.yaml")


def test_brx_keeps_the_original_task_and_camera():
    base = load_config(str(Path(SIM_ROOT) / "configs" / "pick_cube.yaml"))
    cfg = load_config(CONFIG)
    for key in ("scene", "appearance", "perception", "policy", "camera", "camera_intrinsics"):
        assert cfg[key] == base[key]
    assert cfg["mode"] == "single" and cfg["robot"] == "brx20260825"
    assert not cfg["rig"]["arm_mount"]
    original = rig_geometry(base, ["right"])
    geometry = rig_geometry(cfg, ["right"])
    np.testing.assert_allclose(geometry.T_cam_in_world, original.T_cam_in_world, atol=1e-12)
    np.testing.assert_allclose(geometry.T_cam_in_world @ geometry.T_base_in_cam["right"],
                               geometry.T_base_in_world["right"], atol=1e-12)
    np.testing.assert_allclose(geometry.T_base_in_world["right"][:3, 3],
                               cfg["rig"]["base_pose"]["right"]["position"])


def test_brx_rejects_duplicating_the_whole_robot():
    cfg = load_config(CONFIG, {"mode": "bimanual"})
    with pytest.raises(ValueError, match="supports modes"):
        build_scene(cfg, ROBOTS["brx20260825"], get_camera(cfg["camera"]))


@pytest.mark.parametrize("position", [[0, 1], [0, float("nan"), 0]])
def test_invalid_base_pose_fails_before_building(position):
    cfg = load_config(CONFIG, {"rig": {"base_pose": {"right": {"position": position}}}})
    with pytest.raises(ValueError, match="finite position"):
        rig_geometry(cfg, ["right"])


@pytest.fixture
def env():
    if not (Path(ASSETS) / ROBOTS["brx20260825"].xml).is_file():
        pytest.skip("BRX assets not imported; run simulation/scripts/import_brx.py")
    env = ServeBreadSim(load_config(CONFIG), render=False)
    env.reset()
    return env


def test_only_right_arm_and_jaws_move_and_scene_stays_settled(env):
    model = env.m
    assert env.sides == ["right"] and model.nu == 8
    assert len(env.arms["right"].qadr) == 7
    joints = {model.joint(i).name for i in range(model.njnt)}
    assert joints == {"right/" + n for n in env.robot.arm_joints + list(env.robot.gripper_reset_qpos)} | {
        "cube_free", "container_free"}
    assert model.body("right/ArmL08_Link").id > 0
    assert model.body("right/base_link").id > 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rail") == -1
    before = env.d.qpos.copy()
    env.step(round(5 / env.dt))
    assert np.isfinite(env.d.qpos).all() and np.isfinite(env.d.qvel).all()
    np.testing.assert_allclose(env.arms["right"].get_j(), before[env.arms["right"].qadr], atol=1e-4)
    for name in env.obj_names:
        np.testing.assert_allclose(env.object_pos(name)[:2], env.cfg["scene"]["objects"][name]["pose"][:2], atol=0.002)
    for contact in env.d.contact[:env.d.ncon]:
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
        if any(name.startswith("right/") for name in bodies):
            assert contact.dist > -0.001, bodies


def test_brx_camera_pose_and_ik_follow_the_actual_tcp(env):
    arm = env.arms["right"]
    world = env.T_cam_in_world @ arm.get_T_ee_in_cam()
    np.testing.assert_allclose(world, arm.T_base_in_world @ arm.get_T_ee_in_base(), atol=1e-10)
    for xyz in ([0.4, -0.3, 0.017], [0.4, -0.3, 0.12], [0.292460543, 0.109612, 0.14]):
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler("xyz", [180, 0, 180], degrees=True).as_matrix()
        target[:3, 3] = xyz
        q, error = arm.ik(arm.T_world_in_base @ target, arm.get_j(), iters=500)
        actual = arm.T_base_in_world @ arm.fk_raw(q)
        assert error < 0.001
        assert Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() < 0.01
        assert np.all(q >= arm.qlo) and np.all(q <= arm.qhi)


def test_brx_cartesian_servo_and_measured_jaw_direction(env):
    arm = env.arms["right"]
    target = arm.T_base_in_world @ arm.get_T_ee_in_base()
    target[2, 3] -= 0.04
    assert arm.move_p(T_to_p(arm.T_world_in_base @ target), duration=0.8,
                      safe_z_min=env.cfg["control"]["safe_z_min"])
    env.step(round(1.5 / env.dt))
    actual = arm.T_base_in_world @ arm.get_T_ee_in_base()
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 0.002
    assert Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() < 0.01
    assert arm.get_gripper_q() > 0.95
    arm.close_gripper(goal_time=0.5)
    env.step(round(1.0 / env.dt))
    assert arm.get_gripper_q() < 0.03
    q = [env.d.qpos[env.m.jnt_qposadr[env.m.joint("right/" + name).id]]
         for name in ("JawBlock01_Joint", "JawBlock02_Joint")]
    assert q[0] > 0.039 and q[1] < -0.039
    assert abs(sum(q)) < 0.0005
    arm.open_gripper(goal_time=0.5)
    env.step(round(1.0 / env.dt))
    assert arm.get_gripper_q() > 0.95


def test_brx_lifts_with_both_pads_and_releases_inside_container(env):
    """Scripted ground-truth targets validate mechanics, independently of policy quality."""
    from humanego_sim.perception import SideObjects
    from humanego_sim.runner import EpisodeRunner, SideCtx

    arm = env.arms["right"]
    cube = env.object_pos("cube").copy()
    container = env.object_pos("container").copy()

    def move(xyz):
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler("xyz", [180, 0, 180], degrees=True).as_matrix()
        target[:3, 3] = xyz
        assert arm.move_p(T_to_p(arm.T_world_in_base @ target), duration=2,
                          safe_z_min=env.cfg["control"]["safe_z_min"])
        env.step(round(2.3 / env.dt))
        actual = arm.T_base_in_world @ arm.get_T_ee_in_base()
        assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 0.002
        assert Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() < 0.01

    def pad_forces():
        forces = np.zeros(2)
        fingers = [env.m.body("right/" + name).id for name in env.robot.finger_bodies]
        wrench = np.zeros(6)
        for index, contact in enumerate(env.d.contact[:env.d.ncon]):
            bodies = env.m.geom_bodyid[contact.geom]
            if env.obj_bodies["cube"] in bodies:
                for side, body in enumerate(fingers):
                    if body in bodies:
                        mujoco.mj_contactForce(env.m, env.d, index, wrench)
                        forces[side] += abs(wrench[0])
        return forces

    move(cube)
    arm.close_gripper(goal_time=0.5)
    env.step(round(1 / env.dt))
    assert np.all(pad_forces() > 0.1)
    move([cube[0], cube[1], 0.10])
    assert env.object_pos("cube")[2] - cube[2] > 0.025
    assert np.all(pad_forces() > 0.1)
    move([cube[0], cube[1], 0.12])
    move([0.36, -0.13, 0.12])
    # Any safe point in the cavity is a valid drop; avoid the robot's trunk.
    move([container[0] + 0.02, container[1] - 0.04, 0.12])
    arm.open_gripper(goal_time=0.5)
    env.step(round(1.5 / env.dt))
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg, runner.env, runner.task = env.cfg, env, "pick_cube"
    ctx = SideCtx("right", False, SideObjects())
    ctx.hand["has_ever_grasped"] = True
    assert arm.get_gripper_q() > 0.95
    assert runner.evaluate({"right": ctx})["right"]["success"]
