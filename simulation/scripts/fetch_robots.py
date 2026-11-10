# -*- coding: utf-8 -*-
"""
Download robot models that are not shipped with the repo (they are large) from
MuJoCo Menagerie into simulation/humanego_sim/assets/robots/.

    python simulation/scripts/fetch_robots.py franka_panda ur5e
    python simulation/scripts/fetch_robots.py all

The Trossen WidowX AI (the real HumanEgo rig) is committed and needs no download.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.robots import ASSETS, ROBOTS  # noqa: E402

MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"


def sparse_clone(folders, dest):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE, dest], check=True)
    subprocess.run(["git", "-C", dest, "sparse-checkout", "set", *folders], check=True)


def main(names):
    if names == ["all"]:
        names = [n for n, r in ROBOTS.items() if r.fetch]
    jobs = []
    for n in names:
        f = ROBOTS[n].fetch
        if not f:
            print(f"{n}: {ROBOTS[n].setup_command or 'shipped with the repo'}")
            continue
        jobs.append((os.path.dirname(ROBOTS[n].xml).split("/")[-1], f))
        for ex in f.get("extra", []):
            jobs.append((ex["dest"], ex))
    if not jobs:
        return
    with tempfile.TemporaryDirectory() as tmp:
        sparse_clone(sorted({j[1]["menagerie"] for j in jobs}), tmp)
        for dest, f in jobs:
            out = os.path.join(ASSETS, "robots", dest)
            os.makedirs(out, exist_ok=True)
            for item in f["files"]:
                src = os.path.join(tmp, f["menagerie"], item)
                dst = os.path.join(out, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(src, dst)
            print(f"fetched {f['menagerie']} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
