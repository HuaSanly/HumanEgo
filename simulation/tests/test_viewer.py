from types import SimpleNamespace

import mujoco.viewer
import pytest

from humanego_sim.viewer import show_simulation


def test_viewer_syncs_and_cleans_up(monkeypatch):
    class Viewer:
        running = True
        syncs = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def is_running(self):
            return self.running

        def sync(self):
            self.syncs += 1

    viewer = Viewer()
    env = SimpleNamespace(m=object(), d=object(), dt=0, viewer_update=None)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda m, d: viewer)
    with pytest.raises(KeyboardInterrupt):
        with show_simulation(env, True):
            env.viewer_update()
            assert viewer.syncs >= 2
            viewer.running = False
            env.viewer_update()
    assert env.viewer_update is None
    assert viewer.closed


def test_disabled_viewer_does_not_open_window(monkeypatch):
    def fail(*args):
        raise AssertionError("viewer should not be opened")

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fail)
    with show_simulation(object()):
        pass
