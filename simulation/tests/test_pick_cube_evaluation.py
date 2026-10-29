"""Frozen evaluation inputs, independent seeds, resumability and clustered statistics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from humanego_sim import pick_cube_evaluation as evaluation
from humanego_sim.env import SIM_ROOT, load_config
from humanego_sim.pick_cube_sampling import layout_reason, make_trials, validate_trials
from humanego_sim.runner import EpisodeRunner


@pytest.fixture
def inputs(tmp_path):
    cfg = load_config(str(Path(SIM_ROOT) / "configs/pick_cube.yaml"))
    checkpoint = tmp_path / "weights/latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixed test checkpoint")
    (checkpoint.parent / "config.json").write_text("{}")
    (checkpoint.parent / "dataset_stats.json").write_text("{}")
    cfg["policy"]["ckpt"] = str(checkpoint)
    sampling = yaml.safe_load(evaluation.SAMPLING_CONFIG.read_text())
    return cfg, sampling


def test_manifest_separates_scene_environment_and_policy_seeds(inputs):
    cfg, sampling = inputs
    before = copy.deepcopy(cfg)
    manifest = make_trials(cfg, sampling, 6, 7777, [0, 8, 99])
    assert len(manifest["trials"]) == 18 and cfg == before
    assert manifest == make_trials(cfg, sampling, 6, 7777, [0, 8, 99])
    assert manifest["trials"] == make_trials(cfg, sampling, 8, 7777, [0, 8, 99])["trials"][:18]
    for offset in range(0, 18, 3):
        trials = manifest["trials"][offset:offset + 3]
        assert [t["policy_seed"] for t in trials] == [0, 8, 99]
        assert all(t["scene"] == trials[0]["scene"] and t["env_seed"] == trials[0]["env_seed"] for t in trials)
        assert layout_reason(trials[0]["scene"], cfg, sampling) == ""
    other = make_trials(cfg, sampling, 6, 8888, [0, 8, 99])
    assert other["trials"][0]["scene"] != manifest["trials"][0]["scene"]
    validate_trials(manifest, cfg, sampling)


@pytest.mark.parametrize("change", ["overlap", "table", "workspace", "camera"])
def test_invalid_layout_is_rejected_without_running_policy(inputs, change):
    cfg, sampling = inputs
    scene = {name: list(cfg["scene"]["objects"][name]["pose"]) for name in ("cube", "container")}
    if change == "overlap":
        scene["cube"] = scene["container"].copy()
    elif change == "table":
        scene["cube"][0] = cfg["rig"]["table"]["x_max"] + 0.1
    elif change == "workspace":
        sampling["workspace"]["x"][1] = 0.4
    else:
        sampling["camera_margin_px"] = 300
    reason = layout_reason(scene, cfg, sampling)
    assert reason == {"overlap": "objects_overlap", "table": "off_table", "workspace": "outside_workspace",
                      "camera": "outside_camera"}[change]


def test_impossible_bounds_fail_and_rejections_are_reported(inputs):
    cfg, sampling = inputs
    sampling["workspace"]["y"][1] = 0.245
    manifest = make_trials(cfg, sampling, 10, 7777, [0])
    assert manifest["rejections"].get("outside_workspace", 0) > 0
    sampling["max_attempts_per_scene"] = 3
    sampling["workspace"]["x"] = [0.0, 0.1]
    with pytest.raises(ValueError, match="cannot sample scene"):
        make_trials(cfg, sampling, 1, 7777, [0])


@pytest.mark.parametrize("n,seeds", [(0, [0]), (2, []), (2, [0, 0]), (1, [-1])])
def test_invalid_counts_or_noise_seeds(inputs, n, seeds):
    with pytest.raises(ValueError):
        make_trials(*inputs, n, 7777, seeds)


def test_manifest_edits_and_repeat_environment_drift_are_rejected(inputs):
    cfg, sampling = inputs
    manifest = make_trials(cfg, sampling, 1, 7777, [0, 1])
    manifest["trials"][1]["env_seed"] += 1
    with pytest.raises(ValueError, match="identical poses"):
        validate_trials(manifest, cfg, sampling)
    manifest["trials"][0]["trial_id"] = "../../other"
    with pytest.raises(ValueError, match="identity"):
        validate_trials(manifest, cfg, sampling)


def prepare(tmp_path, inputs, n=2, seeds=None):
    cfg, sampling = inputs
    manifest = make_trials(cfg, sampling, n, 7777, seeds or [0, 1])
    root = tmp_path / "study"
    protocol = evaluation.prepare_study(root, cfg, sampling, manifest, "cpu")
    return root, manifest, protocol


def result(root, manifest, protocol, idx, status):
    trial = manifest["trials"][idx]
    row = {"trial": trial, "protocol_sha256": evaluation.digest(protocol), "status": status,
           "success": status == "success", "failure_reason": "error" if status == "error" else "not_grasped"}
    if status != "error":
        row.update(physical={key: status == "success" for key in ("both_pads", "lifted", "secure_grasp", "transported")},
                   duration=10, metrics={"ik_fail": 2}, per_side={"right": {"inside_container": status == "success",
                   "released": True, "settled": True, "release_assist": {"triggered": status == "success"}}})
    directory = root / "episodes" / trial["trial_id"]
    directory.mkdir(parents=True, exist_ok=True)
    evaluation.write_json(directory / "result.json", row)
    return row


def test_freeze_requires_new_output_after_config_weights_or_code_change(tmp_path, inputs, monkeypatch):
    root, manifest, protocol = prepare(tmp_path, inputs)
    cfg, _, loaded, saved = evaluation.load_study(root)
    assert loaded == manifest and saved == protocol and Path(cfg["policy"]["ckpt"]).is_absolute()
    with pytest.raises(ValueError, match="not empty"):
        evaluation.prepare_study(root, *inputs, manifest, "cpu")
    monkeypatch.setattr(evaluation, "code_digest", lambda: "other code")
    with pytest.raises(ValueError, match="code or dependencies"):
        evaluation.load_study(root)
    # Reporting needs neither the original runtime nor checkpoint file.
    Path(cfg["policy"]["ckpt"]).unlink()
    evaluation.load_study(root, verify_runtime=False)
    frozen = root / "config.yaml"
    frozen.write_text(frozen.read_text().replace("max_episode_time: 40.0", "max_episode_time: 39.0"))
    with pytest.raises(ValueError, match="frozen study changed"):
        evaluation.load_study(root, verify_runtime=False)


def test_checkpoint_content_change_is_detected(tmp_path, inputs):
    root, _, _ = prepare(tmp_path, inputs)
    Path(inputs[0]["policy"]["ckpt"]).write_bytes(b"replaced weights at same path")
    with pytest.raises(ValueError, match="checkpoint or training metadata"):
        evaluation.load_study(root)


def test_partial_batch_and_errors_are_not_silently_removed_from_statistics(tmp_path, inputs):
    root, manifest, protocol = prepare(tmp_path, inputs)
    result(root, manifest, protocol, 0, "success")
    result(root, manifest, protocol, 1, "error")
    partial = evaluation.summarize(root, manifest, protocol)
    assert partial["pending"] == 2 and partial["success_rate"] is None
    assert partial["scene_bootstrap95"] is None
    result(root, manifest, protocol, 2, "failure")
    result(root, manifest, protocol, 3, "success")
    summary = evaluation.summarize(root, manifest, protocol)
    assert summary["planned"] == summary["completed"] == 4
    assert summary["success_rate"] == 0.5 and summary["errors"] == 1
    assert summary["per_policy_seed"]["1"]["success_rate"] == 0.5
    assert summary["physical_rates"]["secure_grasp"] == 0.5
    assert summary["scene_bootstrap95"] == [0.5, 0.5]  # each scene has one success out of two
    assert len((root / "results.csv").read_text().splitlines()) == 5
    assert "异常 1" in (root / "report.md").read_text()


def test_result_from_another_trial_cannot_be_reused(tmp_path, inputs):
    root, manifest, protocol = prepare(tmp_path, inputs)
    row = result(root, manifest, protocol, 0, "success")
    row["trial"]["policy_seed"] = 999
    evaluation.write_json(root / "episodes/trial_000000/result.json", row)
    manifest = json.loads((root / "trials.json").read_text())
    with pytest.raises(ValueError, match="does not match"):
        evaluation.read_results(root, manifest, protocol)


def test_complete_resume_does_not_create_workers_or_rerun(tmp_path, inputs, monkeypatch):
    root, manifest, protocol = prepare(tmp_path, inputs, n=1)
    result(root, manifest, protocol, 0, "success")
    result(root, manifest, protocol, 1, "error")
    context = Mock(side_effect=AssertionError("completed trials must not run again"))
    monkeypatch.setattr(evaluation.mp, "get_context", context)
    summary = evaluation.run_study(root, workers=3, device="cpu")
    assert summary["completed"] == 2 and summary["errors"] == 1
    context.assert_not_called()


def test_explicit_error_retry_only_schedules_errors_and_incomplete_trials(tmp_path, inputs, monkeypatch):
    root, manifest, protocol = prepare(tmp_path, inputs)
    result(root, manifest, protocol, 0, "success")
    result(root, manifest, protocol, 1, "error")
    # Trial 2 was interrupted after recording physics but before atomic result publication.
    interrupted = root / "episodes/trial_000002"
    interrupted.mkdir(parents=True)
    (interrupted / "episode.json").write_text("{}")
    scheduled = []

    class Pool:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def imap_unordered(self, function, jobs):
            scheduled.extend(jobs)
            for trial in jobs:
                yield result(root, manifest, protocol, trial["idx"], "failure")

    monkeypatch.setattr(evaluation.mp, "get_context", lambda method: SimpleNamespace(Pool=lambda *a, **kw: Pool()))
    evaluation.run_study(root, workers=2, device="cpu", retry_errors=True)
    assert [t["idx"] for t in scheduled] == [1, 2, 3]


@pytest.mark.parametrize("secure,opening,expected", [(True, 1.0, True), (False, 1.0, False), (True, 0.3, False)])
def test_worker_scores_real_grasp_and_measured_release(tmp_path, inputs, secure, opening, expected):
    root, manifest, protocol = prepare(tmp_path, inputs, n=1, seeds=[0])
    trial = manifest["trials"][0]
    side = {"success": True, "released": True, "inside_container": True, "settled": True}
    value = SimpleNamespace(success=True, per_side={"right": side}, duration=5, wall_time=0.1, metrics={"ik_fail": 0})
    measured = {"both_pads": True, "max_lift_duration": 0.3, "max_secure_duration": 0.2 if secure else 0,
                "max_horizontal": 0.1}
    env = SimpleNamespace(measurements=measured, telemetry=[{"forces": [0, 0]}], renderer=Mock(),
                          arms={"right": SimpleNamespace(get_gripper_q=lambda: opening)})

    def run(**kwargs):
        assert kwargs["seed"] == trial["env_seed"] and kwargs["policy_seed"] == trial["policy_seed"]
        evaluation.write_json(Path(kwargs["log_dir"]) / "episode.json", {"meta": {"result": {
            "success": True, "per_side": {"right": {"success": True}}}}, "frames": [], "policy": []})
        return value

    evaluation.STATE.clear()
    evaluation.STATE.update(root=root, env=env, runner=SimpleNamespace(run=run))
    row = evaluation._run_trial(trial)
    assert row["success"] is expected and row["status"] == ("success" if expected else "failure")
    ep = json.loads((root / "episodes/trial_000000/episode.json").read_text())
    assert ep["meta"]["result"]["success"] is expected and ep["meta"]["result"]["runner_success"] is True


def test_noise_seed_reset_is_independent_of_other_rollouts():
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.policy = SimpleNamespace(noise=torch.zeros(1, 50, 10))
    runner._seed_policy_noise(7)
    first = runner.policy.noise.clone()
    runner._seed_policy_noise(99)
    assert not torch.equal(first, runner.policy.noise)
    runner._seed_policy_noise(7)
    assert torch.equal(first, runner.policy.noise)


def test_runner_explicit_policy_seed_does_not_change_environment_seed(monkeypatch):
    import humanego_sim.runner as module
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg, runner.cc = {"policy": {"seed": 7}}, {}
    runner.env = SimpleNamespace(sides=["right"], reset=Mock())
    runner.policy = SimpleNamespace(noise=torch.zeros(1, 50, 10))
    monkeypatch.setattr(module, "SimPerception", Mock(side_effect=RuntimeError("stop after seed setup")))
    for env_seed in (11, 99):
        with pytest.raises(RuntimeError, match="stop after seed setup"):
            runner.run(seed=env_seed, policy_seed=3, intro=False)
        assert runner._episode_seeds == {"env_seed": env_seed, "policy_seed": 3}
        assert runner.env.rng.random() == np.random.default_rng(env_seed).random()
        expected = torch.randn((1, 50, 10), generator=torch.Generator().manual_seed(3))
        assert torch.equal(runner.policy.noise, expected)
    with pytest.raises(RuntimeError):
        runner.run(seed=11, intro=False)
    assert runner._episode_seeds["policy_seed"] == 18  # keep the legacy base-seed + episode-seed path


def test_cli_preparation_and_checkpoint_free_reporting(tmp_path, inputs):
    cfg, _ = inputs
    config = tmp_path / "task.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "cli_study"
    command = [sys.executable, str(Path(SIM_ROOT) / "scripts/evaluate.py")]
    first = subprocess.run([*command, "--config", str(config), "--n", "2", "--scene-seed", "7777",
                            "--policy-seeds", "0", "1", "--out", str(output), "--prepare-only"],
                           text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    manifest = json.loads((output / "trials.json").read_text())
    assert len(manifest["trials"]) == 4
    resumed = subprocess.run([*command, "--out", str(output), "--resume", "--prepare-only"],
                             text=True, capture_output=True)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((output / "trials.json").read_text()) == manifest
    Path(cfg["policy"]["ckpt"]).unlink()
    reported = subprocess.run([*command, "--out", str(output), "--report-only"], text=True, capture_output=True)
    assert reported.returncode == 0, reported.stderr
    assert json.loads((output / "summary.json").read_text())["pending"] == 4
