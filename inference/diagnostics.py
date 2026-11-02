"""Opt-in per-request files; tensors are copied only in full diagnostic mode."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_active = ContextVar("humanego_trace", default=None)


def current_trace():
    return _active.get()


def trace_path(root, message):
    return (
        Path(root) / message.session_id / str(message.generation) / message.request_id
    )


class RequestTrace:
    def __init__(self, path):
        self.path = Path(path)
        self.write_errors = []
        self.status = "completed"

    def _write(self, name, write):
        path = self.path / name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path)
        except (OSError, ValueError, TypeError, cv2.error):
            self.write_errors.append(name)
            logger.exception("diagnostic_write_failed path=%s", path)

    def json(self, name, value):
        self._write(
            name,
            lambda path: path.write_text(json.dumps(value, indent=2, allow_nan=False)),
        )

    def array(self, name, value):
        self._write(
            name, lambda path: np.save(path, as_array(value), allow_pickle=False)
        )

    def arrays(self, name, **values):
        self._write(
            name,
            lambda path: np.savez_compressed(
                path, **{k: as_array(v) for k, v in values.items()}
            ),
        )

    def image(self, name, image, *, rgb=False):
        value = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image

        def write(path):
            if not cv2.imwrite(str(path), value):
                raise OSError(f"Cannot save image: {path}")

        self._write(name, write)


def as_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def record_request(root, message, config):
    if root is None:
        yield None
        return
    trace = RequestTrace(trace_path(root, message))
    body = message.body
    metadata = {
        "request_id": message.request_id,
        "session_id": message.session_id,
        "generation": message.generation,
        "observation_id": body.get("observation_id"),
        "capture_time_client_monotonic_s": body.get("capture_time"),
        "task": body.get("task"),
        "started_time_ns": time.time_ns(),
        "status": "processing",
    }
    trace.json("request.json", metadata)
    trace.json(
        "config.json",
        {
            "policy": config.policy,
            "camera": config.camera,
            "perception": config.perception,
            "intrinsics": config.intrinsics,
            "extrinsics": config.extrinsics,
        },
    )
    features = body.get("features", {})
    for side in ("left", "right"):
        trace.image(
            f"input/{side}_rgb.png",
            features[f"observation.images.head_{side}"],
            rgb=True,
        )
    trace.array("input/state.npy", features["observation.state"])
    token = _active.set(trace)
    logger.info("diagnostic_request path=%s", trace.path)
    try:
        yield trace
        metadata["status"] = trace.status
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        _active.reset(token)
        metadata.update(
            finished_time_ns=time.time_ns(), write_errors=trace.write_errors
        )
        trace.json("request.json", metadata)
