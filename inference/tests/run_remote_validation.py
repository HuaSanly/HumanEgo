"""Actual policy-server process and official Zenoh client/ChunkRuntime; no motors."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import torch
import yaml
from lerobot.inference import (
    ChunkRuntime,
    ExecutionMode,
    ObservationSnapshot,
    RemoteInferenceConfig,
)
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.protocol import (
    Envelope,
    MessageType,
    ErrorCode,
    ProtocolError,
)


class RecordedRobot:
    """Test-only observation source and sink. send_action records, never actuates."""

    def __init__(self, path):
        data = np.load(path)
        self.features = {
            "observation.images.head_left": data["left"],
            "observation.images.head_right": data["right"],
            "observation.state": data["state"],
        }
        self.actions = []

    def get_observation(self):
        return ObservationSnapshot(
            self.features, time.monotonic(), "pick_cube", observation_id=uuid4().hex
        )

    def send_action(self, action):
        assert action.shape == (10,) and torch.isfinite(action).all()
        self.actions.append(action.clone())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = yaml.safe_load((args.bundle / "server.yaml").read_text())
    config = RemoteInferenceConfig(
        endpoint=server["zenoh"]["listen_endpoints"][0],
        deployment=server["deployment"],
        semantics=server["semantics"],
        chunk_merge="append",
        max_observation_age_s=10,
        handshake_timeout_s=2,
        action_timeout_s=30,
        startup_timeout_s=30,
        refill_seconds=1.65,
    )
    log = (args.output / "server.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_policy_server",
            "--config_path",
            str((args.bundle / "server.yaml").resolve()),
            "--model.repo_or_path",
            str(args.bundle.resolve()),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    client = None
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Policy server exited: see {args.output}/server.log"
                )
            try:
                client = RemoteClient.connect(config)
                break
            except (ConnectionError, TimeoutError, ProtocolError):
                time.sleep(0.2)
        if client is None:
            raise TimeoutError("Policy server did not become ready within 120s")
        caps = client.capabilities

        def admit():
            client.admit(
                features=caps.features,
                action_feature=caps.action_feature,
                semantics=config.semantics,
                action_interval=caps.action_interval,
                mode="chunk",
            )

        bad = replace(caps.features[0], shape=(240, 320, 3))
        try:
            client.admit(
                features=(bad, *caps.features[1:]),
                action_feature=caps.action_feature,
                semantics=config.semantics,
                action_interval=caps.action_interval,
                mode="chunk",
            )
        except ProtocolError as exc:
            assert exc.code == ErrorCode.INCOMPATIBLE
        else:
            raise AssertionError("Wrong schema was admitted")
        admit()
        robot = RecordedRobot(args.observation)
        runtime = ChunkRuntime(
            mode=ExecutionMode.CHUNK,
            action_interval=1 / 30,
            refill_seconds=1.65,
            max_observation_age_s=10,
            action_timeout_s=30,
            startup_timeout_s=30,
            action_starvation_grace_s=3,
            chunk_merge="append",
        )
        runtime.activate()
        accepted = 0
        future = None
        request = None
        latencies = []
        server_durations = []
        first = None
        next_tick = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            deadline = time.monotonic() + 90
            while accepted < args.chunks and time.monotonic() < deadline:
                if future is not None and future.done():
                    result = future.result()
                    assert result.canonical_actions.shape == (50, 10)
                    assert runtime.accept(request, result, task_version=0)
                    latencies.append(time.monotonic() - request.submitted_at)
                    server_durations.append(result.server_durations)
                    if first is None:
                        first = result.canonical_actions.clone()
                    accepted += 1
                    future = None
                if (
                    future is None
                    and runtime.should_request()
                    and accepted < args.chunks
                ):
                    request = runtime.begin(robot.get_observation())
                    assert request is not None
                    future = pool.submit(client.infer, request)
                if time.monotonic() >= next_tick:
                    item = runtime.pop()
                    if item is not None:
                        robot.send_action(item[0])
                    next_tick += 1 / 30
                assert runtime.failure is None, runtime.failure
                time.sleep(0.002)
        assert accepted == args.chunks
        assert len(robot.actions) > 50
        # Invalid payload must fail on the actual server, after a session is admitted.
        obs = robot.get_observation()
        body = client._observation_body(obs)
        body["features"]["observation.state"] = np.zeros(9, np.float32)
        body.update(
            delay=0, model_continuation=None, canonical_continuation=None, cursor=0
        )
        malformed = Envelope(
            MessageType.OBSERVATION,
            client.instance_id,
            client.session_id,
            client.generation,
            uuid4().hex,
            body,
        )
        try:
            client._exchange(malformed, "/obs", client._actions, 30, lambda: False)
        except ProtocolError as exc:
            assert exc.code == ErrorCode.INCOMPATIBLE, (exc.code, exc)
        else:
            raise AssertionError("Malformed wire observation succeeded")
        # Reset increments generation and prevents the old request from producing an action.
        client.control("reset", 1)
        stale = replace(
            malformed,
            generation=0,
            request_id=uuid4().hex,
            body={**body, "features": client._observation_body(obs)["features"]},
        )
        try:
            client._exchange(stale, "/obs", client._actions, 30, lambda: False)
        except ProtocolError as exc:
            assert exc.code == ErrorCode.STALE, exc
        else:
            raise AssertionError("Stale generation produced actions")
        fresh = ChunkRuntime(
            mode=ExecutionMode.CHUNK,
            action_interval=1 / 30,
            refill_seconds=1.65,
            max_observation_age_s=10,
            action_timeout_s=30,
            startup_timeout_s=30,
            chunk_merge="append",
        )
        fresh.activate()
        fresh.generation = 1
        request = fresh.begin(robot.get_observation())
        after = client.infer(request)
        torch.testing.assert_close(after.canonical_actions, first, rtol=0, atol=0)
        assert fresh.accept(request, after, task_version=0)
        # The local official runtime also rejects a result invalidated during an in-flight request.
        invalid = replace(request, request_id=uuid4().hex)
        fresh.invalidate(held=True)
        assert not fresh.accept(invalid, after, task_version=0)
        assert client.close()
        client = None
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=15)
        assert process.returncode == 0
        report = dict(
            chunks=accepted,
            recorded_actions=len(robot.actions),
            action_fps=30,
            request_seconds=latencies,
            server_durations=server_durations,
            outside_worker_seconds=[
                latency - durations["worker"] - durations["queue"]
                for latency, durations in zip(latencies, server_durations, strict=True)
            ],
            initial_scene_s=latencies[0],
            steady_mean_s=float(np.mean(latencies[1:])),
            steady_p95_s=float(np.percentile(latencies[1:], 95)),
            mean_s=float(np.mean(latencies)),
            p95_s=float(np.percentile(latencies, 95)),
            checks=[
                "actual upstream server process",
                "Zenoh raw640 stereo",
                "continuous refill via ChunkRuntime",
                "canonical float32 50x10",
                "wrong admission schema rejected",
                "malformed wire payload rejected",
                "reset deterministic replay",
                "stale generation rejected",
                "invalidated result discarded",
                "session close",
                "SIGTERM stop",
            ],
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if client is not None:
            client.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()


if __name__ == "__main__":
    main()
