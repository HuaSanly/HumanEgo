"""Prop geometry, dataset roles and scene-only export, without policy weights."""
import json
import os

import mujoco
import numpy as np
import pytest
import yaml

from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.live_export import export_live
from humanego_sim.scene_preview import record_scene
from humanego_sim.web_export import export_model, export_rollout
from humanego_sim import web_manifest, web_pack

# Keep primitive-box regression coverage alongside the calibrated default scene.
CONFIG = os.path.join(SIM_ROOT, "configs", "pick_cube.yaml")


@pytest.fixture
def env():
    return ServeBreadSim(load_config(CONFIG, {"scene": {"objects": {"container": {"shape": "box"}}}}), render=False)


def test_default_pick_cube_is_standalone_without_bread_settings(tmp_path):
    with open(CONFIG) as file:
        raw = yaml.safe_load(file)
    cube = load_config(CONFIG)
    standalone = tmp_path / "pick_cube.yaml"
    standalone.write_text(yaml.safe_dump(raw))
    assert load_config(str(standalone)) == cube == raw
    assert "base_config" not in raw and "per_robot" not in raw
    assert set(cube["scene"]) == {"objects"}
    assert not any(key.startswith(("bread_", "plate_")) or key == "soft" for key in cube["physics"])
    assert not {"use_idle_done", "idle_time_limit", "idle_dist_threshold"} & cube["control"].keys()
    assert cube["scene"]["objects"]["container"]["shape"] == "rounded"
    assert cube["scene"]["objects"]["cube"]["pose"] == [0.4, -0.2, 0.0]
    assert 0 < cube["control"]["grasp_threshold"] < 1
    assert cube["control"]["forced_lift_height"] == 0.05
    assert cube["control"]["grasp_assist"]["orientation_world_rpy_deg"] == [180, 0, 180]
    assert not cube["scene_only"] and cube["mode"] == "single"
    with pytest.raises(ValueError, match="tuned profiles are for serve_bread"):
        load_config(CONFIG, {"profile": "tuned"})


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
        assert env.m.body_mass[env.obj_bodies[name]] == pytest.approx(cfg[name]["mass"], abs=1e-6)


def test_scene_recording_and_web_export(env, tmp_path):
    log_dir = str(tmp_path / "run")
    result = record_scene(env, log_dir, duration=0.1)
    assert result["success"] is None and result["per_side"] == {} and not result["evaluated"]
    episode = json.loads((tmp_path / "run" / "episode.json").read_text())
    assert episode["policy"] == [] and len(episode["frames"]) > 1
    model = export_model(env, str(tmp_path), "pick_cube_test")
    assert not model["scene_only"] and model["task"] == "pick_cube"
    assert set(model["objects"]) == {"container", "cube"}
    assert {g["type"] for g in model["geoms"] if g["group"] == 2} == {"box"}
    live_dir = export_live(env, str(tmp_path), "pick_cube_test")
    spec = json.load(open(os.path.join(live_dir, "spec.json")))
    assert not spec["scene_only"] and spec["object_roles"] == {"obj1": ["container"], "obj2": ["cube"]}
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


def test_pick_cube_success_uses_full_container_interior(env):
    from humanego_sim.runner import EpisodeRunner, SideCtx
    from humanego_sim.perception import SideObjects
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg, runner.env, runner.task = env.cfg, env, "pick_cube"
    ctx = SideCtx("right", False, SideObjects())
    env.reset({"container": [0.47, 0, 30]})
    env.place_object("cube", [0.47, 0, 0.15, 1, 0, 0, 0])
    mujoco.mj_forward(env.m, env.d)
    env.step(round(1.5 / env.dt))
    assert not runner.evaluate({"right": ctx})["right"]["success"]  # no grasp occurred
    ctx.hand["has_ever_grasped"] = True
    assert runner.evaluate({"right": ctx})["right"]["success"]
    # Centre is inside, but a corner is on the wall: this is not a completed placement.
    target = env.body_T_world("container")
    pos = target[:3, :3] @ np.array([0.075, 0, -0.011]) + target[:3, 3]
    env.place_object("cube", [*pos, 1, 0, 0, 0])
    mujoco.mj_forward(env.m, env.d)
    assert not runner.evaluate({"right": ctx})["right"]["inside_container"]


@pytest.mark.sim
def test_local_pick_cube_checkpoint_runs_closed_loop(tmp_path):
    from humanego_sim.env import REPO_ROOT
    ckpt = os.path.join(REPO_ROOT, "runs", "pick_cube", "HumanEgo", "latest.pt")
    if not os.path.exists(ckpt):
        pytest.skip("local trained pick_cube checkpoint not present")
    import torch
    from humanego_sim.runner import EpisodeRunner
    cfg = load_config(CONFIG, {"control": {"max_episode_time": 2.0}})
    try:
        env = ServeBreadSim(cfg)
    except Exception as e:
        pytest.skip(f"no OpenGL context: {e!r}")
    try:
        runner = EpisodeRunner(cfg, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        initial = np.asarray(env.initial_joints("right"))
        result = runner.run(log_dir=str(tmp_path), verbose=False, intro=False)
        assert result.n_policy_steps >= 5
        assert np.linalg.norm(env.arms["right"].get_j() - initial) > 0.05
        assert "inside_container" in result.per_side["right"]
        assert "dist_to_plate_center" not in result.per_side["right"]
        ep = json.loads((tmp_path / "episode.json").read_text())
        assert ep["meta"]["policy"]["ckpt"] == cfg["policy"]["ckpt"]
        assert ep["meta"]["task"] == "pick_cube" and not ep["meta"]["scene_only"]
        assert set(ep["frames"][0]["objects"]) == {"container", "cube"}
    finally:
        env.renderer.close()


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
