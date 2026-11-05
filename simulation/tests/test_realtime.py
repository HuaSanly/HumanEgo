"""Real-time physics must keep advancing while policy inference is blocked."""
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.perception import SideObjects
from humanego_sim.realtime import RealtimeInference, remaining_plan
from humanego_sim.runner import EpisodeRunner


def test_one_inflight_job_and_error_propagation():
    started, release = Event(), Event()

    def infer(value):
        started.set()
        assert release.wait(2)
        raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        with RealtimeInference(SimpleNamespace(infer=infer)) as worker:
            worker.submit({"right": {"inputs": (1,)}})
            assert started.wait(2)
            assert worker.poll() is None
            with pytest.raises(RuntimeError, match="already in flight"):
                worker.submit({})
            release.set()
            # Exiting waits for the model and surfaces unconsumed failures.


@pytest.mark.parametrize("age,stride,expected", [(0, 1, 0), (0.3, 1, 3), (0.3, 2, 6), (5, 1, 10)])
def test_delay_trims_pose_and_gripper_together(age, stride, expected):
    raw = list(range(10))
    grasp = np.arange(10)
    poses, commands = remaining_plan(raw, grasp, age, 10, stride)
    assert poses == raw[expected:]
    np.testing.assert_array_equal(commands, grasp[expected:])


@pytest.mark.parametrize("blocked", [None, "muted", "home", "expired"])
def test_runner_advances_physics_while_model_is_blocked(monkeypatch, tmp_path, blocked):
    import humanego_sim.runner as module

    cfg = load_config(SIM_ROOT + "/configs/pick_cube.yaml", {
        "execution_mode": "realtime", "control": {"max_episode_time": 0.4, "policy_period": 10,
            "exec_horizon": 1 if blocked == "expired" else 50, "release_assist": {"enabled": False}}})
    env = ServeBreadSim(cfg, render=False)
    started, release = Event(), Event()
    main_thread = get_ident()
    advancing = []
    contexts, accepted = [], []

    def infer(observation):
        assert get_ident() != main_thread
        started.set()
        assert release.wait(2), "physics loop blocked on inference"
        return {"right": (None, None, np.zeros((50, 1)))}, 1.0 if blocked else 0.0

    policy = SimpleNamespace(frame_mode="camera_frame", action_mode="absolute", single_hand_side="right",
                             noise=torch.zeros(1, 50, 10), infer=infer)
    runner = EpisodeRunner(cfg, policy=policy, env=env)
    objects = SideObjects({}, np.eye(4), {})
    monkeypatch.setattr(module, "SimPerception", lambda *a, **kw: SimpleNamespace(
        estimate_objects=lambda *a, **kw: {"right": objects}, clean_background=lambda: None))
    monkeypatch.setattr(runner, "_perceived_geometry", lambda *a: {})
    def prepare(ctx, bg, t):
        contexts.append(ctx)
        return {"inputs": (np.array([t]),), "t": t, "grasping": False, "latched": None}

    monkeypatch.setattr(runner, "_prepare_side", prepare)

    def finish(ctx, request, result):
        assert get_ident() == main_thread
        accepted.append(request)
        ctx.is_done = result[1] > 0.8
        return ([env.arms["right"].get_p().copy() for _ in range(50)], np.zeros((50, 1)),
                None, np.zeros((50, 3)), 0.0)

    monkeypatch.setattr(runner, "_finish_side", finish)
    step = env.step

    def advancing_step(n=1):
        assert get_ident() == main_thread
        step(n)
        if started.is_set() and not release.is_set() and env.t >= 0.12:
            advancing.append(env.t)
            if blocked == "muted":
                runner._controller.muted["right"] = True
            elif blocked == "home":
                contexts[0].wrap_phase = "home"
                contexts[0].wrap_t0 = env.t
            release.set()

    monkeypatch.setattr(env, "step", advancing_step)
    result = runner.run(log_dir=str(tmp_path), intro=False, save_policy_images=False, verbose=False)
    assert advancing
    assert result.n_policy_steps >= 1
    assert bool(accepted) == (blocked is None)
    assert not contexts[0].is_done
    import json
    log = json.loads((tmp_path / "episode.json").read_text())
    assert log["meta"]["execution_mode"] == "realtime"
    first = log["policy"][0]
    assert first["applied_t"] >= 0.12 and first["t"] == 0
    assert first["latency_wall"] > 0
