"""Request-correlated INFO logs around the pinned official worker and transport."""

from contextvars import ContextVar
import logging
import time

from lerobot.remote_inference import SessionWorker
from lerobot.remote_inference.protocol import MessageType
from lerobot.remote_inference.codec import decode_message
from lerobot.transport import ZenohTransport

from .diagnostics import RequestTrace, record_request, trace_path

logger = logging.getLogger(__name__)
_request = ContextVar("humanego_request", default="request=local")


def request_tag() -> str:
    return _request.get()


def message_tag(message) -> str:
    return (
        f"request={message.request_id} session={message.session_id} "
        f"generation={message.generation}"
    )


class LoggedSessionWorker(SessionWorker):
    def __init__(self, *args, trace_root=None, **kwargs):
        self.trace_root = trace_root
        super().__init__(*args, **kwargs)

    def submit(self, message):
        if message.message_type is MessageType.OBSERVATION:
            features = message.body.get("features")
            shapes = (
                {
                    name: f"{getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}"
                    for name, value in features.items()
                }
                if isinstance(features, dict)
                else "invalid"
            )
            logger.info(
                "observation_received %s observation=%r features=%s",
                message_tag(message),
                message.body.get("observation_id"),
                shapes,
            )
        return super().submit(message)

    def _execute(self, message):
        if message.message_type is not MessageType.OBSERVATION:
            return super()._execute(message)
        token = _request.set(message_tag(message))
        started = time.perf_counter()
        logger.info("inference_started %s", request_tag())
        try:
            with record_request(self.trace_root, message) as trace:
                response = super()._execute(message)
                if trace:
                    if response.message_type is MessageType.ACTION:
                        trace.array(
                            "output/returned_actions.npy",
                            response.body["canonical_actions"],
                        )
                        trace.json(
                            "output/runner_durations.json",
                            response.body.get("server_durations", {}),
                        )
                    else:
                        trace.status = "error_response"
                        trace.json("output/error.json", response.body)
            if response.message_type is MessageType.ACTION:
                logger.info(
                    "action_ready %s shape=%s elapsed_ms=%.1f",
                    request_tag(),
                    response.body["canonical_actions"].shape,
                    (time.perf_counter() - started) * 1000,
                )
            return response
        except Exception:
            logger.exception(
                "inference_failed %s elapsed_ms=%.1f",
                request_tag(),
                (time.perf_counter() - started) * 1000,
            )
            raise
        finally:
            _request.reset(token)


class LoggedZenohTransport(ZenohTransport):
    def __init__(self, config, *, trace_root=None):
        super().__init__(config)
        self.trace_root = trace_root

    def publish(self, key, payload):
        response = decode_message(payload) if key.endswith("/act") else None
        root = getattr(self, "trace_root", None)
        trace = RequestTrace(trace_path(root, response)) if root and response else None
        try:
            super().publish(key, payload)
        except Exception as exc:
            if trace:
                event = {
                    "status": "failed",
                    "error": str(exc),
                    "time_ns": time.time_ns(),
                }
                trace.json(f"publications/{event['time_ns']}.json", event)
                trace.json("publication.json", event)
            logger.exception(
                "reply_publish_failed key=%s %s",
                key,
                message_tag(response) if response else "",
            )
            raise
        if trace:
            event = {
                "status": "published",
                "message_type": response.message_type.value,
                "time_ns": time.time_ns(),
                "bytes": len(payload),
                "delivery_ack": False,
                "server_durations": response.body.get("server_durations"),
                "error_code": response.body.get("code"),
                "error_message": response.body.get("message"),
            }
            trace.json(f"publications/{event['time_ns']}.json", event)
            trace.json("publication.json", event)
            if response.message_type is MessageType.ACTION:
                trace.array(
                    "output/published_actions.npy", response.body["canonical_actions"]
                )
        if response is not None:
            if response.message_type is MessageType.ACTION:
                logger.info(
                    "action_published %s shape=%s bytes=%d durations_s=%s delivery_ack=false",
                    message_tag(response),
                    response.body["canonical_actions"].shape,
                    len(payload),
                    response.body.get("server_durations"),
                )
            elif response.message_type is MessageType.ERROR:
                logger.warning(
                    "error_published %s code=%s message=%s",
                    message_tag(response),
                    response.body.get("code"),
                    response.body.get("message"),
                )
