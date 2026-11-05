"""Optional native MuJoCo viewer for a running simulation."""
from contextlib import contextmanager
import time


@contextmanager
def show_simulation(env, enabled=False):
    if not enabled:
        yield
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        last_step = time.monotonic()
        last_sync = 0.0

        def update():
            nonlocal last_step, last_sync
            if not viewer.is_running():
                raise KeyboardInterrupt("MuJoCo viewer closed")
            now = time.monotonic()
            delay = env.dt - (now - last_step)
            if delay > 0:
                time.sleep(delay)
            last_step = time.monotonic()
            if last_step - last_sync >= 1 / 30:
                viewer.sync()
                last_sync = last_step

        env.viewer_update = update
        try:
            viewer.sync()
            yield
        finally:
            env.viewer_update = None
