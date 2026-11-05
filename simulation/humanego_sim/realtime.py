"""One in-flight policy job; only immutable observation inputs cross threads."""
from concurrent.futures import ThreadPoolExecutor
import time


class RealtimeInference:
    def __init__(self, policy):
        self.policy = policy
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-policy")
        self.future = None
        self.requests = None
        self.started = None
        self.last_step = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # A running model call cannot be cancelled. Finish it before reusing the
        # policy in the next episode, or destroying its CUDA resources.
        self.pool.shutdown(wait=True, cancel_futures=True)
        if exc_type is None and self.future is not None and not self.future.cancelled():
            self.future.result()

    @property
    def busy(self):
        return self.future is not None

    def submit(self, requests):
        if self.busy:
            raise RuntimeError("a policy job is already in flight")
        self.requests = requests
        self.started = time.monotonic()
        self.future = self.pool.submit(self._infer, requests)

    def _infer(self, requests):
        return {side: self.policy.infer(*request["inputs"]) for side, request in requests.items()}

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        results = self.future.result()
        packet = self.requests, results, time.monotonic() - self.started
        self.future = self.requests = None
        return packet

    def pace(self, dt):
        """Pace headless real-time runs; a visible viewer already paces physics."""
        delay = dt - (time.monotonic() - self.last_step)
        if delay > 0:
            time.sleep(delay)
        self.last_step = time.monotonic()


def remaining_plan(raw, grasp, age, control_freq, step_stride):
    """Align an observation-time trajectory with the controller's current tick."""
    skipped = max(0, int((age + 1e-9) * control_freq) * step_stride)
    return raw[skipped:], grasp[skipped:]
