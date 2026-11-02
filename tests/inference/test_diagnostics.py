"""Detailed traces retain inputs and partial failures without changing inference."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from lerobot.remote_inference import SessionWorker
from lerobot.remote_inference.codec import encode_message
from lerobot.remote_inference.protocol import Envelope, MessageType
from lerobot.transport import ZenohTransport
from lerobot_policy_humanego.diagnostics import (
    current_trace,
    record_request,
    trace_path,
)
from lerobot_policy_humanego.runtime_logging import (
    LoggedSessionWorker,
    LoggedZenohTransport,
)


def input_message():
    image = np.full((480, 640, 3), [123, 45, 67], np.uint8)
    return Envelope(
        MessageType.OBSERVATION,
        "instance",
        "session",
        0,
        "request-1",
        {
            "features": {
                "observation.images.head_left": image,
                "observation.images.head_right": image.copy(),
                "observation.state": np.arange(10, dtype=np.float32),
            },
            "capture_time": 1.0,
            "observation_id": "frame-1",
            "task": "pick_cube",
        },
    )


def config():
    return SimpleNamespace(
        policy={}, camera={}, perception={}, intrinsics={}, extrinsics={}
    )


@pytest.mark.parametrize("failed", [False, True])
def test_trace_keeps_exact_input_and_partial_failure(tmp_path, failed):
    worker = LoggedSessionWorker.__new__(LoggedSessionWorker)
    worker.trace_root = tmp_path
    worker.runner = SimpleNamespace(policy=SimpleNamespace(serving_config=config()))
    request = input_message()
    actions = np.zeros((50, 10), np.float32)

    def execute(self, message):
        current_trace().array("model/mask.npy", torch.tensor([True, False]))
        if failed:
            raise ValueError("localization failed")
        return message.reply(MessageType.ACTION, {"canonical_actions": actions})

    with patch.object(SessionWorker, "_execute", execute):
        if failed:
            with pytest.raises(ValueError, match="localization failed"):
                worker._execute(request)
        else:
            worker._execute(request)
    assert current_trace() is None
    folder = trace_path(tmp_path, request)
    metadata = json.loads((folder / "request.json").read_text())
    assert metadata["status"] == ("failed" if failed else "completed")
    np.testing.assert_array_equal(
        np.load(folder / "input/state.npy"), np.arange(10, dtype=np.float32)
    )
    image = cv2.cvtColor(
        cv2.imread(str(folder / "input/left_rgb.png")), cv2.COLOR_BGR2RGB
    )
    np.testing.assert_array_equal(
        image, request.body["features"]["observation.images.head_left"]
    )
    assert np.load(folder / "model/mask.npy").dtype == np.bool_
    if failed:
        assert metadata["error"] == "localization failed"
        assert not (folder / "output/returned_actions.npy").exists()
    else:
        np.testing.assert_array_equal(
            np.load(folder / "output/returned_actions.npy"), actions
        )


def test_trace_records_actual_transport_output_and_failure(tmp_path):
    transport = LoggedZenohTransport.__new__(LoggedZenohTransport)
    transport.trace_root = tmp_path
    request = input_message()
    actions = np.full((5, 10), 0.6, np.float32)
    response = request.reply(MessageType.ACTION, {"canonical_actions": actions})
    with patch.object(ZenohTransport, "publish"):
        transport.publish("session/act", encode_message(response))
    folder = trace_path(tmp_path, request)
    np.testing.assert_array_equal(
        np.load(folder / "output/published_actions.npy"), actions
    )
    assert (
        json.loads((folder / "publication.json").read_text())["delivery_ack"] is False
    )
    # Different request, so a send failure cannot leave a prior success artifact.
    failed = Envelope(
        MessageType.ACTION, "instance", "session", 0, "request-2", response.body
    )
    with patch.object(ZenohTransport, "publish", side_effect=OSError("closed")):
        with pytest.raises(OSError):
            transport.publish("session/act", encode_message(failed))
    failed_folder = trace_path(tmp_path, failed)
    assert (
        json.loads((failed_folder / "publication.json").read_text())["status"]
        == "failed"
    )
    assert not (failed_folder / "output/published_actions.npy").exists()


def test_disabled_mode_does_not_create_files(tmp_path):
    with record_request(None, input_message(), None) as trace:
        assert trace is None
        assert current_trace() is None
    assert not list(tmp_path.iterdir())


def test_disk_error_does_not_break_processing(tmp_path):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied")
    with record_request(blocked, input_message(), config()) as trace:
        assert trace.write_errors
        trace.array("data.npy", np.zeros(1))
    assert current_trace() is None


def test_duplicate_error_preserves_action_publication_history(tmp_path):
    transport = LoggedZenohTransport.__new__(LoggedZenohTransport)
    transport.trace_root = tmp_path
    request = input_message()
    response = request.reply(
        MessageType.ACTION, {"canonical_actions": np.zeros((2, 10), np.float32)}
    )
    error = request.reply(
        MessageType.ERROR, {"code": "duplicate", "message": "Duplicate request"}
    )
    with patch.object(ZenohTransport, "publish"):
        transport.publish("session/act", encode_message(response))
        transport.publish("session/act", encode_message(error))
    folder = trace_path(tmp_path, request)
    events = [
        json.loads(p.read_text()) for p in (folder / "publications").glob("*.json")
    ]
    assert {event["message_type"] for event in events} == {"action", "error"}
    assert (
        json.loads((folder / "publication.json").read_text())["error_code"]
        == "duplicate"
    )


def test_stereo_output_survives_object_localization_failure(tmp_path):
    from lerobot_policy_humanego.perception import Perception
    from test_geometry_scene import config as scene_config, observation

    masks = {key: np.ones((480, 640), bool) for key in ("obj1", "obj2", "arms")}
    perception = Perception(
        scene_config(),
        segmentation=SimpleNamespace(segment=lambda *a, **kw: masks),
        inpainter=object(),
    )
    depth = SimpleNamespace(
        disparity_px=np.ones((480, 640), np.float32),
        depth_m=np.ones((480, 640), np.float32),
        valid_mask=np.ones((480, 640), bool),
    )
    points = np.zeros((8, 3), np.float32)
    perception.stereo = SimpleNamespace(
        estimate=lambda frame: depth,
        points_from_mask=iter([points, ValueError("bad points")]),
    )
    results = perception.stereo.points_from_mask

    def locate(*args):
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    perception.stereo.points_from_mask = locate
    request = input_message()
    with pytest.raises(ValueError, match="bad points"):
        with record_request(tmp_path, request, config()):
            perception.observe(observation())
    folder = trace_path(tmp_path, request)
    np.testing.assert_array_equal(
        np.load(folder / "stereo/depth.npz")["depth_m"], depth.depth_m
    )
    saved = np.load(folder / "stereo/object_points.npz")
    assert saved.files == ["obj1"]
    np.testing.assert_array_equal(saved["obj1"], points)


def test_lama_input_survives_inference_failure(tmp_path):
    from lerobot_policy_humanego.perception import Inpainter

    inpainter = Inpainter.__new__(Inpainter)
    inpainter.opts = {"mask_dilation": 9, "input_size_hw": [512, 512]}
    inpainter.session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="image"),
            SimpleNamespace(name="mask"),
        ],
        run=lambda *args: (_ for _ in ()).throw(RuntimeError("GPU failed")),
    )
    image = np.full((480, 640, 3), [50, 100, 200], np.uint8)
    mask = np.ones((480, 640), bool)
    request = input_message()
    with pytest.raises(RuntimeError, match="GPU failed"):
        with record_request(tmp_path, request, config()):
            inpainter.inpaint(image, mask, np.zeros_like(mask))
    folder = trace_path(tmp_path, request)
    saved = np.load(folder / "lama/inputs.npz")
    np.testing.assert_allclose(
        saved["image_bgr"][0, :, 0, 0], np.array([50, 100, 200]) / 255
    )
    assert saved["mask"].all()
    assert not (folder / "lama/output.npy").exists()
