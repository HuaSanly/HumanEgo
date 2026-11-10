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


def test_brx_keeps_the_original_policy_camera_and_restores_container_height():
    base = load_config(str(Path(SIM_ROOT) / "configs" / "pick_cube.yaml"))
    cfg = load_config(CONFIG)
    for key in ("appearance", "perception", "policy", "camera", "camera_intrinsics"):
        assert cfg[key] == base[key]
    for name, obj in base["scene"]["objects"].items():
        actual = cfg["scene"]["objects"][name]
        excluded = {"pose", "size"} if name == "container" else {"pose"}
        assert {k: v for k, v in actual.items() if k not in excluded} == {
            k: v for k, v in obj.items() if k not in excluded}
        if name == "container":
            assert actual["size"][:2] == obj["size"][:2]
            assert actual["size"][2] == pytest.approx(0.09647649800453617)
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


@pytest.mark.parametrize("surface", [0.0, 0.10, -0.10])
def test_perceived_grasp_height_is_relative_to_the_table(surface):
    from types import SimpleNamespace
    from humanego_sim.runner import EpisodeRunner
    from humanego_sim.perception import SimPerception

    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = load_config(CONFIG)
    runner.cc, runner.task = runner.cfg["control"], "pick_cube"
    runner.env = SimpleNamespace(T_cam_in_world=np.eye(4), meta={"table_surface_z": surface})
    size = runner.cfg["scene"]["objects"]["cube"]["size"][2]
    top = surface + size
    points = np.array([[0.3, 0, top], [0.33, 0, top], [0.3, 0.03, top], [0.33, 0.03, top]])
    sensed = SimpleNamespace(objs_meta={"obj2": {"T_obj_in_cam_static": np.eye(4), "kpts_local": points}})
    geometry = runner._perceived_geometry("right", sensed)
    assert geometry["bread"]["c"][2] == pytest.approx(surface + size / 2)
    perception = SimPerception.__new__(SimPerception)
    perception.env = runner.env
    near_table = np.array([0.3, 0.0, surface + 0.002])
    above_table = np.array([0.3, 0.0, surface + 0.020])
    filtered = perception._reject_off_object([near_table, above_table], margin=0.005)
    assert filtered[0] is None
    np.testing.assert_allclose(filtered[1], above_table)


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
    np.testing.assert_allclose(env.body_T_world("right/base_link")[:3, 3], [-0.14639, 0.32001, -0.32104])
    np.testing.assert_allclose(env.body_T_world("right/base_link")[:3, :3], np.eye(3), atol=1e-10)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rail") == -1
    ready = env.arms["right"].T_base_in_world @ env.arms["right"].get_T_ee_in_base()
    tilt = np.rad2deg(np.arccos(np.clip(-ready[2, 2], -1, 1)))
    assert 30 < tilt < 60
    assert np.linalg.norm(ready[:2, 3] - env.object_pos("cube")[:2]) > 0.10
    before = env.d.qpos.copy()
    env.step(round(5 / env.dt))
    assert np.isfinite(env.d.qpos).all() and np.isfinite(env.d.qvel).all()
    np.testing.assert_allclose(env.arms["right"].get_j(), before[env.arms["right"].qadr], atol=1e-4)
    for name in env.obj_names:
        np.testing.assert_allclose(env.object_pos(name)[:2], env.cfg["scene"]["objects"][name]["pose"][:2], atol=0.002)
        assert env.object_lowest_z(name) == pytest.approx(env.meta["table_surface_z"], abs=0.002)
    assert model.geom_pos[model.geom("floor").id, 2] == -0.76
    assert env.body_T_world("table")[2, 3] == pytest.approx(0.10)
    for contact in env.d.contact[:env.d.ncon]:
        bodies = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
        if any(name.startswith("right/") for name in bodies):
            assert contact.dist > -0.001, bodies


def test_fixed_torso_remains_vertical(env):
    bodies = [env.m.body("right/" + name).id for name in
              ("FoldingModularJoint02_Link", "FoldingModularJoint03_Link", "Trunk_Link")]
    for lower, upper in zip(bodies, bodies[1:]):
        delta = env.d.xpos[upper] - env.d.xpos[lower]
        np.testing.assert_allclose(delta, [0, 0, 0.3], atol=3e-6)
    # The trunk's local -Y is vertical in the original upright CAD pose.
    rotation = env.d.xmat[bodies[-1]].reshape(3, 3)
    np.testing.assert_allclose(rotation @ [0, -1, 0], [0, 0, 1], atol=1e-5)


def test_head_camera_frames_the_separated_props(env):
    assert np.linalg.norm(env.object_pos("cube")[:2] - env.object_pos("container")[:2]) > 0.35
    camera = env.m.camera("right/head_view").id
    assert env.m.cam_bodyid[camera] == env.m.body("right/EYEM_Link").id
    rotation = env.d.cam_xmat[camera].reshape(3, 3)
    forward = -rotation[:, 2]
    assert forward[2] < -0.7  # physical optical axis points down at the table
    optical = rotation @ np.diag([1, -1, -1])  # MuJoCo -> OpenCV camera axes
    focal = env.H / (2 * np.tan(np.deg2rad(env.m.cam_fovy[camera] / 2)))
    for name, obj in env.cfg["scene"]["objects"].items():
        half = np.asarray(obj["size"]) / 2
        if name == "container":
            half[:2] += obj["rim"]
            half[1] += 0.016  # include the handles
        corners = np.array([[x, y, z] for x in (-half[0], half[0])
                            for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        world = corners @ env.body_T_world(name)[:3, :3].T + env.object_pos(name)
        points = (world - env.d.cam_xpos[camera]) @ optical
        assert np.all(points[:, 2] > 0)
        pixels = points[:, :2] / points[:, 2:] * focal + [env.W / 2, env.H / 2]
        assert np.all(pixels > 20) and np.all(pixels < [env.W - 20, env.H - 20])


def test_raised_table_layout_and_browser_reset_share_height(env, tmp_path):
    import json
    import yaml
    from humanego_sim.live_export import export_live
    from humanego_sim.pick_cube_sampling import layout_reason

    with open(Path(SIM_ROOT) / "configs" / env.cfg["evaluation"]["sampling"]) as file:
        sampling = yaml.safe_load(file)
    assert layout_reason(env.default_scene(), env.cfg, sampling) == ""
    folder = Path(export_live(env, str(tmp_path), "brx_raised_table"))
    spec = json.loads((folder / "spec.json").read_text())
    assert spec["table_surface_z"] == pytest.approx(0.10)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, obj in env.cfg["scene"]["objects"].items():
        # MJCF serializes positions to six decimal places.
        assert data.xpos[model.body(name).id, 2] == pytest.approx(0.10 + obj["size"][2] / 2 + 0.0008, abs=1e-6)


def test_brx_camera_pose_and_ik_follow_the_actual_tcp(env):
    arm = env.arms["right"]
    world = env.T_cam_in_world @ arm.get_T_ee_in_cam()
    np.testing.assert_allclose(world, arm.T_base_in_world @ arm.get_T_ee_in_base(), atol=1e-10)
    cube, container = env.object_pos("cube"), env.object_pos("container")
    surface = env.meta["table_surface_z"]
    for xyz in (cube, [*cube[:2], surface + 0.12], [*container[:2], surface + 0.12]):
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
    surface = env.meta["table_surface_z"]

    def move(xyz):
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler("xyz", [180, 0, 180], degrees=True).as_matrix()
        target[:3, 3] = xyz
        start = (arm.T_base_in_world @ arm.get_T_ee_in_base())[:3, 3]
        # Short Cartesian commands keep IK on a continuous branch while the
        # upright seven-axis arm crosses the workspace, as in closed-loop control.
        steps = max(1, int(np.ceil(np.linalg.norm(target[:3, 3] - start) / 0.02)))
        for fraction in np.linspace(1 / steps, 1, steps):
            waypoint = target.copy()
            waypoint[:3, 3] = start + fraction * (target[:3, 3] - start)
            assert arm.move_p(T_to_p(arm.T_world_in_base @ waypoint), duration=2 / steps,
                              safe_z_min=env.cfg["control"]["safe_z_min"])
            env.step(round(2 / steps / env.dt))
        env.step(round(0.3 / env.dt))
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
    move([cube[0], cube[1], surface + 0.10])
    assert env.object_pos("cube")[2] - cube[2] > 0.025
    assert np.all(pad_forces() > 0.1)
    move([cube[0], cube[1], surface + 0.12])
    midpoint = (cube[:2] + container[:2]) / 2
    # Cross the wrist's limited workspace at a reachable transport height.
    move([*midpoint, surface + 0.10])
    move([*container[:2], surface + 0.12])
    arm.open_gripper(goal_time=0.5)
    env.step(round(1.5 / env.dt))
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg, runner.env, runner.task = env.cfg, env, "pick_cube"
    ctx = SideCtx("right", False, SideObjects())
    ctx.hand["has_ever_grasped"] = True
    assert arm.get_gripper_q() > 0.95
    assert runner.evaluate({"right": ctx})["right"]["success"]
