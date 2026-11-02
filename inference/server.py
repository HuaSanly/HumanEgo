"""Official LeRobot serving lifecycle with HumanEgo per-request console logs."""

import logging
import signal

from lerobot.configs import parser
from lerobot.remote_inference import PolicyServer, ServerConfig, load_deployment
from lerobot.utils import init_logging, register_third_party_plugins

from .runtime_logging import LoggedSessionWorker, LoggedZenohTransport

logger = logging.getLogger(__name__)


@parser.wrap()
def serve(cfg: ServerConfig) -> None:
    init_logging(console_level=cfg.log_level)
    logger.info(
        "server_loading deployment=%s model=%s device=%s log_level=%s",
        cfg.deployment,
        cfg.model.repo_or_path,
        cfg.model.device,
        cfg.log_level,
    )
    runner, identity = load_deployment(cfg)
    worker = LoggedSessionWorker(
        runner,
        deployment=cfg.deployment,
        artifact_identity=identity,
        semantics=cfg.semantics,
        action_deadline_s=cfg.execution.action_deadline_s,
        language_deadline_s=cfg.language.deadline_s,
        idle_timeout_s=cfg.execution.idle_timeout_s,
        max_input_chars=cfg.language.max_input_chars,
        max_output_chars=cfg.language.max_output_chars,
    )
    server = PolicyServer(worker, LoggedZenohTransport(cfg.zenoh))
    logger.info(
        "server_warmed deployment=%s execution_steps=%d action_hz=%.1f",
        cfg.deployment,
        runner.capabilities.execution_steps,
        1 / runner.capabilities.action_interval,
    )
    previous_handlers = {
        sig: signal.signal(
            sig, lambda signum, _: server.stop(reason=signal.Signals(signum).name)
        )
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        server.serve()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    register_third_party_plugins()
    serve()


if __name__ == "__main__":
    main()
