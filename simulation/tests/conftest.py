# -*- coding: utf-8 -*-
"""Shared fixtures for the simulation test suite.

    pytest simulation/tests                 # everything that can run here
    pytest simulation/tests -m "not sim"    # CPU-only subset (what CI runs)

Markers: `sim` = needs the released checkpoint and an OpenGL context (EGL / OSMesa);
those tests skip themselves when either is missing.
"""
import os
import sys

import pytest

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SIM)
if SIM not in sys.path:
    sys.path.insert(0, SIM)
# scripts/ last: scripts/autotune.py must not shadow the autotune package
if os.path.join(SIM, "scripts") not in sys.path:
    sys.path.append(os.path.join(SIM, "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")

from humanego_sim.robots import ASSETS, ROBOTS  # noqa: E402

CKPT = os.path.join(REPO, "checkpoints", "serve_bread", "latest.pt")


def robot_available(name: str) -> bool:
    return os.path.exists(os.path.join(ASSETS, ROBOTS[name].xml))


def robots_with_assets(mode="bimanual"):
    """Trossen ships with the repo; Franka / UR5e appear after scripts/fetch_robots.py."""
    return [pytest.param(n, marks=pytest.mark.skipif(not robot_available(n), reason=f"{n} assets not fetched"))
            for n, robot in ROBOTS.items() if mode in robot.supported_modes]


@pytest.fixture(scope="session")
def sim_root():
    return SIM
