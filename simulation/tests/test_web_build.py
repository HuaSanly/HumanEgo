# -*- coding: utf-8 -*-
"""Web build: manifest schema / generated notes, packed rollouts (bit-exact), GLB packing, deploy plan."""
import copy
import json
import os
import shutil

import numpy as np
import pytest

from humanego_sim import profiles, web_manifest, web_pack

SIM = profiles.SIM_ROOT
WEB = os.path.join(SIM, "web")


# ---------------------------------------------------------------------------- packed rollouts
def synthetic_rollout(T=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(T) / 30.0 - 1.2, 3)
    q = np.round(np.cumsum(rng.normal(0, 0.01, (T, 12)), 0), 3)
    q[60:, 3] = np.round(q[60:, 3] + 40.0, 3)                         # a jump that needs int32 deltas
    q[:, 4] = np.round(np.linspace(0.0, 20.0, T), 3)                   # steady motion: int16 deltas
    soft = np.round(np.cumsum(rng.normal(0, 1e-3, (T, 7)), 0), 4)
    frames = np.hstack([t[:, None], q, soft]).tolist()
    r = [0, 0] + [rng.integers(-40, 40, 27).tolist() for _ in range(5)] + [0]
    deform = {"bread_right": {"t": [round(0.1 * i, 3) for i in range(len(r))], "r": r}}
    policy = []
    for i in range(10):
        e = {"t": round(0.3 * i, 3), "sides": {}}
        for s in (["right", "left"] if i > 2 else ["right"]):
            e["sides"][s] = {"traj": np.round(rng.uniform(-0.5, 0.8, (26, 3)), 3).tolist(), "traj_step": 2,
                             "grasp": np.round(rng.uniform(0, 1, 12), 2).tolist(),
                             "gmax": round(float(rng.uniform(0, 1)), 2), "done": round(float(rng.uniform(0, 1)), 3),
                             "grasping": bool(i % 2)}
        policy.append(e)
    policy[4]["sides"]["left"]["traj"] = []
    return {"id": "x_000", "model": "x", "fps": 30, "frames": frames, "policy": policy, "policy_video": None,
            "deform": deform, "policy_period": 0.3, "result": {"success": True, "per_side": {}},
            "scene": {"plate": [0.47, 0.0, 0.0]}, "extra": {}, "kpts": {}, "appear": {"plate": -1.2},
            "look_ahead": 45}


def test_pack_rollout_is_bit_exact():
    ro = synthetic_rollout()
    data = web_pack.pack_rollout(ro)
    back = web_pack.unpack_rollout(data)
    assert web_pack._same(ro, back)
    # floats come back as the very same doubles JSON parsing gives
    assert all(a == b for fa, fb in zip(ro["frames"], back["frames"]) for a, b in zip(fa, fb))
    hdr_len = int.from_bytes(data[8:12], "little")
    hdr = json.loads(data[12:12 + hdr_len])
    assert set(hdr["pack"]) == {"frames", "deform", "policy"}            # everything went binary
    assert {c["w"] for c in hdr["pack"]["frames"]["cols"]} >= {1, 2, 4}
    assert len(data) < 0.5 * len(json.dumps(ro, separators=(",", ":")))


def test_pack_rollout_keeps_unknown_sections_as_json():
    ro = synthetic_rollout()
    ro["policy"][0]["sides"]["right"]["new_field"] = 1                 # a schema the packer does not know
    ro["frames"][5][3] = 0.1234567891                                  # not rounded -> no exact integer form
    back = web_pack.unpack_rollout(web_pack.pack_rollout(ro))
    assert web_pack._same(ro, back)


@pytest.mark.skipif(not os.path.isdir(os.path.join(WEB, "data", "rollouts")), reason="no web build")
def test_built_rollouts_round_trip():
    import glob
    fs = sorted(glob.glob(os.path.join(WEB, "data", "rollouts", "*.json")))[:6]
    if not fs:
        pytest.skip("no JSON rollouts in the build")
    for jp in fs:
        ro = json.load(open(jp))
        assert web_pack._same(ro, web_pack.unpack_rollout(web_pack.pack_rollout(ro))), jp


# ---------------------------------------------------------------------------- GLB packing
def _box_scene(path, textured=False):
    trimesh = pytest.importorskip("trimesh")
    sc = trimesh.Scene()
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.05))
    box.apply_translation((0.3, -0.1, 0.02))
    V, F = box.vertices[box.faces].reshape(-1, 3), np.arange(3 * len(box.faces)).reshape(-1, 3)
    a = trimesh.Trimesh(V, F, vertex_normals=np.repeat(box.face_normals, 3, 0), process=False)   # flat-shaded
    ball = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    ball.apply_translation((0.0, 0.2, 0.1))
    b = trimesh.Trimesh(ball.vertices, ball.faces, vertex_normals=ball.vertex_normals, process=False)  # smooth
    if textured:
        from PIL import Image
        img = Image.fromarray((np.random.default_rng(0).uniform(0, 255, (64, 64, 3))).astype(np.uint8))
        uv = np.random.default_rng(1).uniform(0, 1, (len(b.vertices), 2))
        b.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    sc.add_geometry(a, node_name="box", geom_name="box")
    sc.add_geometry(b, node_name="ball", geom_name="ball")
    sc.export(path)
    return {"box": a, "ball": b}


def _world_positions(path, node_name):
    js, bin_ = web_pack._read_glb(path)
    node = next(n for n in js["nodes"] if n.get("name") == node_name)
    p = js["meshes"][node["mesh"]]["primitives"][0]
    P = web_pack._read_accessor(js, bin_, p["attributes"]["POSITION"]).astype(np.float64)
    s = np.asarray(node.get("scale", [1, 1, 1]))
    t = np.asarray(node.get("translation", [0, 0, 0]))
    I = web_pack._read_accessor(js, bin_, p["indices"]).astype(int)
    return P * s + t, I, p, js


def test_quantize_glb(tmp_path):
    path = str(tmp_path / "robot.glb")
    meshes = _box_scene(path)
    rep = web_pack.quantize_glb(path)
    assert rep["flat_meshes"] == 1 and rep["bytes_out"] < rep["bytes_in"]
    js, _ = web_pack._read_glb(path)
    assert "KHR_mesh_quantization" in js["extensionsRequired"]
    for name, m in meshes.items():
        P, I, prim, js = _world_positions(path, name)
        # every original vertex is reproduced within the int16 step (uniform scale -> isotropic error)
        d = np.linalg.norm(m.vertices[:, None, :] - P[None, :, :], axis=2).min(1)
        assert d.max() < 2e-5, name
        assert I.max() < len(P)
        tri = P[I.reshape(-1, 3)]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        assert area == pytest.approx(m.area, rel=1e-3)
        acc = js["accessors"][prim["attributes"]["POSITION"]]
        assert acc["componentType"] == 5122 and acc["normalized"] and len(acc["min"]) == 3
        if name == "box":
            assert "NORMAL" not in prim["attributes"]            # flat: rebuilt in the viewer
        else:
            assert js["accessors"][prim["attributes"]["NORMAL"]]["componentType"] == 5120
    assert web_pack.quantize_glb(path) == {"skipped": True}       # idempotent


def test_webp_glb_keeps_float_positions(tmp_path):
    path = str(tmp_path / "bread.glb")
    _box_scene(path, textured=True)
    js0, bin0 = web_pack._read_glb(path)
    rep = web_pack.webp_glb(path)
    js, bin_ = web_pack._read_glb(path)
    assert "EXT_texture_webp" in js["extensionsRequired"]
    assert all(im["mimeType"] == "image/webp" for im in js["images"])
    for m0, m in zip(js0["meshes"], js["meshes"]):
        a0 = web_pack._read_accessor(js0, bin0, m0["primitives"][0]["attributes"]["POSITION"])
        a1 = web_pack._read_accessor(js, bin_, m["primitives"][0]["attributes"]["POSITION"])
        assert a1.dtype == np.float32 and np.array_equal(a0, a1)    # the viewer deforms these on the CPU
    assert rep["bytes_out"] > 0


# ---------------------------------------------------------------------------- manifest
def _configs():
    cfgs = []
    for robot in ("trossen_wxai", "ur5e"):
        rolls = []
        for i in range(4):
            ps = {"right": {"success": i != 1}, "left": {"success": i % 2 == 0}}
            rolls.append({"id": f"{robot}_realsense_d405_bimanual_{i:03d}",
                          "scene": {"plate": [0.47, 0.0, 0.0], "bread_right": [0.45, -0.25, 95.0],
                                    "bread_left": [0.45, 0.25, 85.123456]},
                          "per_side": ps, "success": all(v["success"] for v in ps.values())})
        cfgs.append({"robot": robot, "camera": "realsense_d405", "mode": "bimanual",
                     "model": f"{robot}_realsense_d405_bimanual", "live": False,
                     "profile": profiles.profile_info("tuned", robot), "rollouts": rolls})
    return cfgs


def test_manifest_numbers_come_from_data():
    man = web_manifest.assemble(_configs(), {"trossen_wxai": "Trossen", "ur5e": "UR5e"},
                                {"realsense_d405": "D405"},
                                {"robot": "trossen_wxai", "camera": "realsense_d405", "mode": "bimanual"}, None)
    assert web_manifest.validate(man) == []
    row = man["summary"][0]
    assert (row["right"], row["left"], row["both"]) == (0.75, 0.5, 0.5)
    for robot in ("trossen_wxai", "ur5e"):
        t = profiles.load_tuned(robot)
        assert web_manifest.pct(t["held_out"]["per_arm"]) in man["summary_note"]
        assert t["autotune_run"] in man["summary_note"]
        r = next(r for r in man["summary"] if r["robot"] == robot)
        first = t.get("final") or t["held_out"]          # the reserved test set leads when it has been run
        assert r["held_out"][0]["per_arm"] == first["per_arm"] and r["profile"] == "tuned"
    assert sum(r.get("featured", False) for r in man["configs"][0]["rollouts"]) == 1
    assert man["configs"][0]["rollouts"][0]["scene"]["bread_left"][2] == 85.1235      # rounded for size


def test_manifest_validation_catches_errors(tmp_path):
    man = web_manifest.assemble(_configs(), {"trossen_wxai": "Trossen", "ur5e": "UR5e"}, {"realsense_d405": "D405"},
                                {"robot": "trossen_wxai", "camera": "realsense_d405", "mode": "bimanual"}, None)
    bad = copy.deepcopy(man)
    bad["configs"][0]["rollouts"][1]["success"] = True                  # not all sides served
    bad["configs"][1]["rollouts"][0]["per_side"].pop("left")
    bad["default"]["robot"] = "franka_panda"
    errs = web_manifest.validate(bad)
    assert len(errs) == 3, errs
    # with a web dir every referenced file must exist
    errs = web_manifest.validate(man, str(tmp_path))
    assert any("missing" in e for e in errs)


def test_adding_scene_preview_keeps_policy_config_and_results(tmp_path, monkeypatch):
    import build_web
    from humanego_sim.env import load_config
    monkeypatch.setattr(build_web, "WEB", str(tmp_path))
    (tmp_path / "data").mkdir()
    bread = _configs()[0]
    objects = load_config(os.path.join(SIM, "configs", "pick_cube.yaml"))["scene"]["objects"]
    preview = {"robot": bread["robot"], "camera": bread["camera"], "mode": bread["mode"],
               "task": "pick_cube", "scene_only": True, "model": "pick_cube_preview", "objects": objects,
               "rollouts": [{"id": "pick_cube_preview_000", "success": None, "per_side": {},
                             "scene": {n: o["pose"] for n, o in objects.items()}}]}
    before = build_web.write_manifest([bread])
    after = build_web.write_manifest([preview])
    assert len(after["configs"]) == 2
    assert after["summary"] == before["summary"]
    assert after["default"]["task"] == "serve_bread"
    assert web_manifest.validate(after) == []


def test_tuning_section_from_run(tmp_path):
    run = tmp_path / "soft_vX"
    (run / "iterations").mkdir(parents=True)
    for i in range(2):
        json.dump({"iteration": i, "candidates": [{"stats": {"per_side_success_rate": 0.5 + 0.1 * i, "score": 0.5}}],
                   "decision": {"action": "continue"}}, open(run / "iterations" / f"iter_{i:03d}.json", "w"))
    json.dump({"trials": [{"name": "production", "validation": {"per_side_success_rate": 0.41}}]},
              open(run / "final.json", "w"))
    tuned = dict(profiles.load_tuned("trossen_wxai"), autotune_run="soft_vX")
    t = web_manifest.tuning_section(tuned, str(tmp_path))
    assert [b["scores"] for b in t["batches"]] == [[0.5], [0.6]]
    h = tuned.get("final") or tuned["held_out"]     # the reserved test set once it has been run
    assert web_manifest.pct(h["per_arm"]) in t["note"] and "41%" not in t["note"]
    assert web_manifest.tuning_section(dict(tuned, autotune_run="nope"), str(tmp_path)) is None


@pytest.mark.skipif(not os.path.exists(os.path.join(WEB, "data", "manifest.json")), reason="no web build")
def test_built_manifest_is_valid():
    man = json.load(open(os.path.join(WEB, "data", "manifest.json")))
    assert web_manifest.validate(man, WEB) == []


def test_deploy_plan_ships_only_what_the_site_loads(tmp_path, monkeypatch):
    import deploy_web
    web = tmp_path / "web"
    for d in ("data/rollouts", "data/models", "assets/textures", "assets/robots", "js", "css", "tools"):
        (web / d).mkdir(parents=True)
    man = web_manifest.assemble(_configs()[:1], {"trossen_wxai": "Trossen"}, {"realsense_d405": "D405"},
                                {"robot": "trossen_wxai", "camera": "realsense_d405", "mode": "bimanual"}, None)
    for r in man["configs"][0]["rollouts"]:
        r["fmt"] = "bin"
        for ext in (".json", ".bin", "_policy.mp4", "_policy.mp4.packed"):
            (web / "data" / "rollouts" / (r["id"] + ext)).write_text("x")
    (web / "data" / "rollouts" / "stale_999.bin").write_text("x")
    json.dump({"geoms": [], "soft": {}, "materials": {"m": {"map": "assets/textures/t.webp"}}},
              open(web / "data" / "models" / "trossen_wxai_realsense_d405_bimanual.json", "w"))
    for f in ("assets/textures/t.png", "assets/textures/t.webp", "index.html", "js/app.js", "tools/serve.py"):
        (web / f).write_text("x")
    json.dump(man, open(web / "data" / "manifest.json", "w"))
    files = deploy_web.plan(str(web))
    assert "assets/textures/t.webp" in files and "assets/textures/t.png" not in files
    assert not any(f.endswith((".json", ".packed")) and "/rollouts/" in f for f in files)
    assert sum(f.endswith(".bin") for f in files) == 4 and "data/rollouts/stale_999.bin" not in files
    assert not any(f.startswith("tools/") for f in files)


def test_readme_results_table_is_generated():
    """README 'Results' == configs/tuned/*.json (regenerate: python -m humanego_sim.web_manifest)."""
    readme = open(os.path.join(SIM, "README.md")).read()
    block = readme.split("<!-- results:start -->\n", 1)[1].split("\n<!-- results:end -->", 1)[0]
    assert block == web_manifest.readme_results()
