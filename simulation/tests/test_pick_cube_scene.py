"""Prop geometry, dataset roles and scene-only export, without policy weights."""
import json
import os

import mujoco
import numpy as np
import pytest

from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.live_export import export_live
from humanego_sim.scene_preview import record_scene
from humanego_sim.web_export import export_model, export_rollout
from humanego_sim import web_manifest, web_pack

CONFIG = os.path.join(SIM_ROOT, "configs", "pick_cube.yaml")


@pytest.fixture
def env():
    return ServeBreadSim(load_config(CONFIG), render=False)


def test_rig_and_control_inherit_original_defaults():
    original, cube = load_config(), load_config(CONFIG)
    for key in ("rig", "robot", "camera", "mode", "policy", "control", "perception"):
        assert cube[key] == original[key]
    assert cube["scene_only"]
    assert original["task"] == "serve_bread" and "scene_only" not in original


@pytest.mark.parametrize("mode", ["single", "bimanual"])
def test_reset_and_roles(mode):
    env = ServeBreadSim(load_config(CONFIG, {"mode": mode}), render=False)
    scene = {"container": [0.50, 0.06, 25], "cube": [0.42, -0.25, 40]}
    env.reset(scene)
    assert set(env.obj_names) == {"container", "cube"}
    assert not env.soft
    assert env.sides == (["right"] if mode == "single" else ["right", "left"])
    assert not any("adhesion" in (env.m.actuator(i).name or "") for i in range(env.m.nu))
    for name, pose in scene.items():
        np.testing.assert_allclose(env.object_pos(name)[:2], pose[:2], atol=0.002)
        assert abs(env.object_lowest_z(name)) < 0.002
    masks = {"container": np.array([[255, 0]], dtype=np.uint8), "cube": np.array([[0, 255]], dtype=np.uint8)}
    mapped = env.object_role_masks(masks)
    np.testing.assert_array_equal(mapped["obj1"], masks["container"])
    np.testing.assert_array_equal(mapped["obj2"], masks["cube"])


def test_cube_falls_into_open_container(env):
    env.reset({"container": [0.47, 0, 30]})
    env.place_object("cube", [0.47, 0, 0.18, 1, 0, 0, 0])
    mujoco.mj_forward(env.m, env.d)
    env.step(round(1.5 / env.dt))
    cfg = env.cfg["scene"]["objects"]
    # The block rests on the bottom inside the cavity, below the upper rim.
    assert env.object_pos("cube")[2] == pytest.approx(cfg["container"]["wall"] + cfg["cube"]["size"][2] / 2, abs=0.002)
    assert env.object_lowest_z("cube") > 0.002
    assert np.isfinite(env.d.qpos).all()
    for name in env.obj_names:
        assert env.m.body_mass[env.obj_bodies[name]] == pytest.approx(cfg[name]["mass"])


def test_scene_recording_and_web_export(env, tmp_path):
    log_dir = str(tmp_path / "run")
    result = record_scene(env, log_dir, duration=0.1)
    assert result["success"] is None and result["per_side"] == {} and not result["evaluated"]
    episode = json.loads((tmp_path / "run" / "episode.json").read_text())
    assert episode["policy"] == [] and len(episode["frames"]) > 1
    model = export_model(env, str(tmp_path), "pick_cube_test")
    assert model["scene_only"] and model["task"] == "pick_cube"
    assert set(model["objects"]) == {"container", "cube"}
    assert {g["type"] for g in model["geoms"] if g["group"] == 2} == {"box"}
    live_dir = export_live(env, str(tmp_path), "pick_cube_test")
    spec = json.load(open(os.path.join(live_dir, "spec.json")))
    assert spec["scene_only"] and spec["object_roles"] == {"obj1": ["container"], "obj2": ["cube"]}
    physics = mujoco.MjModel.from_xml_path(os.path.join(live_dir, "scene.xml"))
    assert physics.nq == env.m.nq
    ro = export_rollout(env, log_dir, str(tmp_path), "pick_cube_test_000")
    assert ro["policy"] == [] and ro["policy_video"] is None
    assert web_pack._same(ro, web_pack.unpack_rollout(web_pack.pack_rollout(ro)))
    cfg = {"robot": env.robot.name, "camera": env.cam.name, "mode": "bimanual", "task": "pick_cube",
           "model": "pick_cube_test", "scene_only": True, "live": True,
           "objects": env.cfg["scene"]["objects"],
           "rollouts": [{"id": ro["id"], "scene": ro["scene"], "success": None, "per_side": {}}]}
    default = {k: cfg[k] for k in ("robot", "camera", "mode", "task")}
    man = web_manifest.assemble([cfg], {env.robot.name: "Trossen"}, {env.cam.name: "D405"}, default, None)
    assert web_manifest.validate(man) == []
    assert man["summary"] == []


@pytest.mark.sim
def test_rendered_rgb_depth_and_masks():
    try:
        env = ServeBreadSim(load_config(CONFIG))
    except Exception as e:
        pytest.skip(f"no OpenGL context: {e!r}")
    try:
        env.reset()
        rgb, depth, masks = env.render_rgb(), env.render_depth(noise=False), env.render_segmentation()
        assert rgb.shape == (env.H, env.W, 3)
        assert depth.shape == (env.H, env.W) and np.isfinite(depth).all()
        assert all(np.count_nonzero(mask) > 20 for mask in masks.values())
        roles = env.object_role_masks(masks)
        assert set(roles) == {"obj1", "obj2"}
    finally:
        env.renderer.close()
