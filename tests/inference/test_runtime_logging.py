"""Logs describe received requests and actual transport publication boundaries."""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.remote_inference import SessionWorker
from lerobot.remote_inference.protocol import Envelope, MessageType
from lerobot.remote_inference.codec import encode_message
from lerobot.transport import ZenohTransport
from lerobot_policy_humanego.runtime_logging import (
    LoggedSessionWorker,
    LoggedZenohTransport,
    request_tag,
)


def message(kind=MessageType.OBSERVATION, body=None):
    return Envelope(kind, "instance", "session", 2, "request-7", body or {})


def test_receive_log_precedes_admission(caplog):
    worker = LoggedSessionWorker.__new__(LoggedSessionWorker)
    received = message(
        body={
            "features": {"observation.state": np.zeros(9, np.float32)},
            "observation_id": "frame\n7",
        }
    )
    with (
        caplog.at_level(logging.INFO),
        patch.object(SessionWorker, "submit", return_value="rejected"),
    ):
        assert worker.submit(received) == "rejected"
    assert "observation_received request=request-7" in caplog.text
    assert "(9,)/float32" in caplog.text
    assert "frame\\n7" in caplog.text
    assert "frame\n7" not in caplog.text
    assert "action_ready" not in caplog.text


@pytest.mark.parametrize("failed", [False, True])
def test_context_follows_request_and_is_cleared_on_failure(caplog, failed):
    worker = LoggedSessionWorker.__new__(LoggedSessionWorker)
    worker.trace_root = None
    response = message(
        MessageType.ACTION, {"canonical_actions": np.zeros((3, 10), np.float32)}
    )

    def execute(self, request):
        assert "request=request-7 session=session generation=2" == request_tag()
        if failed:
            raise ValueError("test failure")
        return response

    with (
        caplog.at_level(logging.INFO),
        patch.object(SessionWorker, "_execute", execute),
    ):
        if failed:
            with pytest.raises(ValueError, match="test failure"):
                worker._execute(message())
        else:
            assert worker._execute(message()) is response
    assert request_tag() == "request=local"
    assert ("inference_failed" if failed else "action_ready") in caplog.text
    assert "action_published" not in caplog.text


@pytest.mark.parametrize("failed", [False, True])
def test_action_publication_is_logged_only_after_transport_success(caplog, failed):
    transport = LoggedZenohTransport.__new__(LoggedZenohTransport)
    response = message(
        MessageType.ACTION, {"canonical_actions": np.zeros((3, 10), np.float32)}
    )
    payload = encode_message(response)
    with (
        caplog.at_level(logging.INFO),
        patch.object(
            ZenohTransport, "publish", side_effect=OSError("closed") if failed else None
        ),
    ):
        if failed:
            with pytest.raises(OSError, match="closed"):
                transport.publish("session/act", payload)
        else:
            transport.publish("session/act", payload)
    assert ("action_published" in caplog.text) is not failed
    assert ("reply_publish_failed" in caplog.text) is failed
    if not failed:
        assert "delivery_ack=false" in caplog.text
        assert "request=request-7 session=session generation=2" in caplog.text
