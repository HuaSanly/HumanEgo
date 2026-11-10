# -*- coding: utf-8 -*-
"""
Quick end-to-end check of the simulator install (~1 min on a GPU):
for every robot whose assets are present, build the scene, run perception and a
few seconds of the closed loop with the released policy.

    python simulation/scripts/smoke_test.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.env import load_config  # noqa: E402
from humanego_sim.robots import ASSETS, ROBOTS  # noqa: E402
from humanego_sim.runner import EpisodeRunner, load_policy  # noqa: E402


def main():
    policy = None
    ok = True
    for name, r in ROBOTS.items():
        if "bimanual" not in r.supported_modes:
            print(f"[skip] {name}: use its task config (simulation/configs/pick_cube_brx.yaml)")
            continue
        if not os.path.exists(os.path.join(ASSETS, r.xml)):
            print(f"[skip] {name}: assets not fetched (python simulation/scripts/fetch_robots.py {name})")
            continue
        cfg = load_config(None, {"robot": name, "control": {"max_episode_time": 4.0}})
        try:
            policy = policy or load_policy(cfg)
            t0 = time.time()
            res = EpisodeRunner(cfg, policy=policy).run(seed=0, verbose=False, intro=False)
            print(f"[ok]   {name}: {res.n_policy_steps} re-plans, {res.duration:.1f} s sim in {time.time() - t0:.1f} s")
        except Exception as e:   # report and keep going
            ok = False
            print(f"[FAIL] {name}: {e!r}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
