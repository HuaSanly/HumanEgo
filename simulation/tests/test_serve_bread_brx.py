"""BRX serves the unchanged bread layout; raised tables keep task semantics."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.robots import ASSETS, ROBOTS
from humanego_sim.runner import EpisodeRunner

CONFIG = str(Path(SIM_ROOT) / "configs" / "serve_bread_brx.yaml")


def test_brx_preserves_the_production_task_and_camera_view():
    from humanego_sim.scene import rig_geometry

    original = load_config(None, {"mode": "single"})
    cfg = load_config(CONFIG)
    assert cfg["task"] == "serve_bread" and cfg["robot"] == "brx20260825"
    assert cfg["mode"] == "single" and cfg["profile"] == "production"
    for key in ("scene", "physics", "perception", "policy", "camera"):
        assert cfg[key] == original[key]
    for key, value in original["control"].items():
        if key != "safe_z_min":
            assert cfg["control"][key] == value, key
    for name in ("grasp_assist", "release_assist", "withdrawal"):
        assert not cfg["control"][name]["enabled"]
    old = rig_geometry(original, ["right"])
    new = rig_geometry(cfg, ["right"])
    np.testing.assert_allclose(new.T_cam_in_world[:3, :3], old.T_cam_in_world[:3, :3])
    np.testing.assert_allclose(new.T_cam_in_world[:3, 3] - [0, 0, 0.20], old.T_cam_in_world[:3, 3])
    np.testing.assert_allclose(new.T_cam_in_world @ new.T_base_in_cam["right"], new.T_base_in_world["right"], atol=1e-12)


@pytest.mark.parametrize("model", ["soft", "rigid"])
@pytest.mark.parametrize("surface", [0.0, 0.20, -0.10])
def test_bread_and_plate_start_above_the_table_before_and_after_reset(model, surface):
    cfg = load_config(None, {"mode": "single", "rig": {"table": {"height": 0.76 + surface}},
                             "physics": {"bread_model": model}})
    env = ServeBreadSim(cfg, render=False)
    mujoco.mj_forward(env.m, env.d)
    for reset in (False, True):
        if reset:
            env.reset(settle=0)
        assert env.object_pos("plate")[2] == pytest.approx(surface + 0.0005, abs=1e-6)
        assert surface - 0.001 < env.object_lowest_z("bread_right") < surface + 0.002
    assert env.m.geom_pos[env.m.geom("floor").id, 2] == -0.76


@pytest.fixture(scope="module")
def brx_env():
    if not (Path(ASSETS) / ROBOTS["brx20260825"].xml).is_file():
        pytest.skip("BRX assets not imported; run simulation/scripts/import_brx.py")
    env = ServeBreadSim(load_config(CONFIG), render=False)
    env.reset()
    return env


def test_brx_layout_is_settled_collision_free_and_reachable(brx_env):
    env = brx_env
    assert env.sides == ["right"] and len(env.arms["right"].qadr) == 7
    arm = env.arms["right"]
    for contact in env.d.contact[:env.d.ncon]:
        bodies = [env.m.body(int(env.m.geom_bodyid[g])).name for g in contact.geom]
        if any(name.startswith("right/") for name in bodies):
            assert contact.dist > -0.001, bodies
    for name in env.obj_names:
        seed = arm.get_j()
        low = env.object_lowest_z(name) if name in env.soft else env.object_pos(name)[2]
        assert abs(low - env.meta["table_surface_z"]) < 0.002
        np.testing.assert_allclose(env.object_pos(name)[:2], env.default_scene()[name][:2], atol=0.002)
        for clearance in (0.026, 0.12):
            target = np.eye(4)
            # Both locations admit a downward pinch; the wrist turns across the plate.
            yaw = 180 if name == "bread_right" else 270
            target[:3, :3] = Rotation.from_euler("xyz", [180, 0, yaw], degrees=True).as_matrix()
            target[:3, 3] = [*env.object_pos(name)[:2], env.meta["table_surface_z"] + clearance]
            q, error = arm.ik(arm.T_world_in_base @ target, seed, iters=500)
            assert error < 0.002, (name, clearance, error)
            actual = arm.T_base_in_world @ arm.fk_raw(q)
            assert Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() < 0.01
            seed = q


@pytest.mark.parametrize("surface", [0.0, 0.20, -0.10])
@pytest.mark.parametrize("on_plate", [True, False])
def test_success_and_failure_classification_use_table_relative_height(surface, on_plate):
    from humanego_sim.evaluate import classify_failure

    plate = np.eye(4)
    bread = np.eye(4)
    bread[:3, 3] = [0, 0, surface + 0.04]
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.task = "serve_bread"
    runner.env = SimpleNamespace(
        meta={"table_surface_z": surface, "objects": {"plate": {"R": 0.15}}},
        body_T_world=lambda name: plate if name == "plate" else bread,
        object_lowest_z=lambda name: surface + (0.009 if on_plate else 0.0008),
        arms={"right": SimpleNamespace(gripper_cmd_closed=lambda: False)},
    )
    runner._bread_max_z = {"right": surface + 0.04}
    ctx = SimpleNamespace(hand={"has_ever_grasped": True}, done_reason="idle_over_plate")
    result = runner.evaluate({"right": ctx})["right"]
    assert result["success"] is on_plate
    assert classify_failure(result) == ("success" if on_plate else "grasp_missed")


@pytest.mark.parametrize("surface", [0.0, 0.20, -0.10])
def test_perceived_place_height_adds_the_table_offset_once(surface):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = load_config(CONFIG)
    runner.cc = {"grasp_assist": {"enabled": True}}
    runner.task, runner.anchor_key = "serve_bread", "obj1"
    runner.env = SimpleNamespace(T_cam_in_world=np.eye(4), meta={"table_surface_z": surface})
    rim = np.array([[0.4, 0.0, surface + 0.022], [0.5, 0.0, surface + 0.022], [0.45, 0.1, surface + 0.022]])
    top = rim.copy()
    top[:, 2] = surface + 0.04
    objects = SimpleNamespace(objs_meta={
        key: {"T_obj_in_cam_static": np.eye(4), "kpts_local": points}
        for key, points in (("obj1", top), ("obj2", rim))
    })
    assert runner._perceived_geometry("right", objects)["place_z"] == pytest.approx(surface + 0.0285)
