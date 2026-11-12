# -*- coding: utf-8 -*-
"""
Build the serve_bread object assets for MuJoCo (OBJ + PNG + convex collision
parts) and for the web viewer (GLB).

Sources
-------
* croissant      procedural model of the straight rolled croissant used in the human
                 demos and robot rollouts (scripts/bread_gen.py)
* bun            procedural model of the laminated sugar danish loaf of the bimanual
                 rollouts (scripts/bread_gen.py)
* plate          procedural lathe (light-blue dinner plate)
* tabletop       procedural pale maple butcher block (like the real rig's table)

Every bread gets a dense textured visual mesh, a coarse closed surface for the soft-body
flex (``<name>_flex.obj``), convex parts for the rigid fallback, and a PBR GLB for the web.

Real sizes come from the 20 triangulated contour keypoints per object in the
61 released serve_bread recordings (camtriangulator_results.json):
    croissant  max extent 12.8 cm (inner-eroded contour; ~13.8 cm outer)
    plate      diameter   23.3 cm (inner-eroded contour; ~24.5 cm outer)
and, for the danish loaf (only in the robot rollouts), from the rollout video against
the 24.5 cm plate: ~14 x 7.4 x 6 cm.

Run:  python simulation/scripts/build_assets.py
"""
import json
import os
import shutil
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
SRC = os.path.join(SIM, "assets_src")
OUT = os.path.join(SIM, "humanego_sim", "assets", "objects")
WEB = os.path.join(SIM, "web", "assets", "objects")

# Y-up (glTF) -> Z-up (MuJoCo): rotate +90 deg about X
YUP_TO_ZUP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def _export_mujoco_parts(mesh, name, n_hulls=6):
    """Convex parts for the rigid-bread fallback (physics.bread_model: rigid)."""
    d = os.path.join(OUT, name)
    parts = []
    try:
        import coacd
        cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        res = coacd.run_coacd(cm, threshold=0.05, max_convex_hull=n_hulls, preprocess_mode="auto",
                              resolution=2000, mcts_iterations=100)
        for v, f in res:
            parts.append(trimesh.Trimesh(v, f).convex_hull)
    except Exception as e:  # pragma: no cover
        print(f"[assets] CoACD unavailable ({e}); falling back to a single hull")
        parts = [mesh.convex_hull]
    for i, p in enumerate(parts):
        p.export(os.path.join(d, f"{name}_col{i}.obj"))
    return len(parts)


def _export_glb(mesh, name, tex_paths=None, color=None, roughness=0.8, metallic=0.0, tex_size=1024):
    os.makedirs(WEB, exist_ok=True)
    m = mesh.copy()
    if tex_paths:
        imgs = {}
        for k, p in tex_paths.items():
            im = Image.open(p).convert("RGB")
            if max(im.size) > tex_size:
                k = tex_size / max(im.size)
                im = im.resize((int(im.size[0] * k), int(im.size[1] * k)), Image.LANCZOS)
            imgs[k] = im
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=imgs.get("diff"),
            normalTexture=imgs.get("nor"),
            metallicRoughnessTexture=imgs.get("arm"),
            occlusionTexture=imgs.get("arm"),
            roughnessFactor=1.0, metallicFactor=1.0 if "arm" in imgs else 0.0)
        m.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=mat)
    elif color is not None:
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=color, roughnessFactor=roughness,
                                                  metallicFactor=metallic)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
    # glTF is Y-up: rotate back so three.js can use a Z-up parent consistently
    m.apply_transform(np.linalg.inv(YUP_TO_ZUP))
    m.export(os.path.join(WEB, f"{name}.glb"))
    sys.path.insert(0, SIM)
    from humanego_sim.glb_tools import jpeg_textures, set_pbr
    jpeg_textures(os.path.join(WEB, f"{name}.glb"), max_size=tex_size)   # PNG -> JPEG for the web
    if tex_paths:   # bread is not metal (and JPEG chroma bleeding would leak into the packed metalness channel)
        set_pbr(os.path.join(WEB, f"{name}.glb"), metallic=0.0, roughness=1.0)


# ---------------------------------------------------------------------------
def build_bread(name: str):
    """Procedural bread -> MuJoCo visual OBJ + texture, flex surface, convex parts, web GLB."""
    sys.path.insert(0, HERE)
    from bread_gen import build
    b = build(name)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    Vv, Fv, UV = b["vis"]
    col, rgh, nrm = b["tex"]
    uv_up = np.c_[UV[:, 0], 1.0 - UV[:, 1]]               # trimesh / OBJ convention (v up)
    vis = trimesh.Trimesh(Vv, Fv, visual=trimesh.visual.TextureVisuals(uv=uv_up), process=False)
    vis.export(os.path.join(d, f"{name}.obj"), include_texture=False)
    Image.fromarray(col).save(os.path.join(d, f"{name}.png"))
    np.savez_compressed(os.path.join(d, f"{name}_vis.npz"), V=Vv, F=Fv, UV=UV)   # exact arrays for the skin
    Vf, Ff = b["flex"]
    flex = trimesh.Trimesh(Vf, Ff, process=False)
    flex.export(os.path.join(d, f"{name}_flex.obj"))
    # finer collision skin (physics.soft.surface: fine): small finger pads cannot slip between vertices
    from bread_gen import SPECS, tube_mesh
    sp = SPECS[name]()
    Vf2, Ff2, _ = tube_mesh(sp, 40, 26, with_relief=False)
    lo, hi = Vf2.min(0), Vf2.max(0)
    Vf2 = (Vf2 - (lo + hi) / 2) / (hi - lo) * np.asarray(sp.size)
    trimesh.Trimesh(Vf2, Ff2, process=False).export(os.path.join(d, f"{name}_flex_fine.obj"))
    for f in os.listdir(d):
        if f.startswith(f"{name}_col"):
            os.remove(os.path.join(d, f))
    n = _export_mujoco_parts(flex, name)
    # web: base colour + tangent-space normal + packed AO/roughness/metal
    tmp = os.path.join(SRC, name)
    os.makedirs(tmp, exist_ok=True)
    paths = {"diff": os.path.join(tmp, "diff.png"), "nor": os.path.join(tmp, "nor.png"),
             "arm": os.path.join(tmp, "arm.png")}
    Image.fromarray(col).save(paths["diff"])
    Image.fromarray(nrm).save(paths["nor"])
    arm = np.stack([np.full_like(rgh, 255), rgh, np.zeros_like(rgh)], -1)
    Image.fromarray(arm).save(paths["arm"])
    _export_glb(vis, name, paths, tex_size=2048)
    with open(os.path.join(d, f"{name}_meta.json"), "w") as f:
        json.dump({"size": b["size"].tolist(), "vis_verts": int(len(Vv)), "flex_verts": int(len(Vf))}, f, indent=1)
    print(f"[assets] {name}: size {np.round(b['size'], 3)}  visual {len(Vv)} v  flex {len(Vf)} v  hulls={n}")


def plate_profile(R=0.1225, H=0.022, r_well=0.080, r_foot=0.072, t=0.0045, foot_h=0.004):
    """(r, z) polyline of a dinner plate cross-section, outer surface then inner."""
    # top surface from centre to rim
    top = [(0.0, foot_h + t), (r_well, foot_h + t)]
    for a in np.linspace(0, 1, 10)[1:]:
        r = r_well + (R - r_well) * a
        z = foot_h + t + (H - foot_h - t) * (a ** 1.6)
        top.append((r, z))
    # rim lip
    top.append((R + 0.002, H - 0.0015))
    # underside back to centre
    bot = []
    for a in np.linspace(1, 0, 10):
        r = r_well + (R - r_well) * a
        z = foot_h + (H - foot_h - t) * (a ** 1.6)
        bot.append((r - 0.001, z))
    bot += [(r_foot + 0.004, foot_h), (r_foot, 0.0), (r_foot - 0.004, 0.0), (r_foot - 0.008, foot_h), (0.0, foot_h)]
    return np.array(top + bot)


def build_plate(R=0.1225, H=0.030):   # rim 2.9 +- 0.4 cm above the table (real video, SLAM 3.5 cm)
    prof = plate_profile(R=R, H=H)
    # trimesh.creation.revolve wants (r, z) with a closed polyline; build manually
    n_seg = 96
    rings = []
    for (r, z) in prof:
        ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        rings.append(np.stack([r * np.cos(ang), r * np.sin(ang), np.full(n_seg, z)], 1))
    V = np.concatenate(rings, 0)
    F = []
    n_r = len(prof)
    for i in range(n_r - 1):
        for j in range(n_seg):
            a = i * n_seg + j
            b = i * n_seg + (j + 1) % n_seg
            c = (i + 1) * n_seg + j
            d = (i + 1) * n_seg + (j + 1) % n_seg
            F += [[a, c, b], [b, c, d]]
    mesh = trimesh.Trimesh(V, np.array(F), process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    color = [0.60, 0.69, 0.86, 1.0]  # light periwinkle-blue plastic plate (real rollouts)
    d = os.path.join(OUT, "plate")
    os.makedirs(d, exist_ok=True)
    mesh.export(os.path.join(d, "plate.obj"))
    _export_glb(mesh, "plate", color=color, roughness=0.18)
    # collision primitives (box ring + base disc) are generated in the MJCF builder
    meta = {"R": R, "H": H, "r_well": 0.080, "t": 0.0045, "foot_h": 0.004, "rgba": color}
    with open(os.path.join(d, "plate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[assets] plate: R={R} H={H} verts={len(mesh.vertices)}")
    return mesh


def make_butcher_block(px=2048, tile_m=0.5, strip_m=0.0455, seed=3):
    """Pale maple butcher-block tile (tile_m x tile_m): glued strips running along the
    table's width, each with its own tone and straight fine grain, satin finish."""
    rng = np.random.default_rng(seed)
    n_strip = int(round(tile_m / strip_m))
    y = np.arange(px)
    x = np.arange(px)
    strip = (y * n_strip) // px                                  # strip index per row
    tones = rng.normal(0, 1, n_strip)
    base_a = np.array([224, 205, 174]) / 255.0
    base_b = np.array([203, 176, 140]) / 255.0
    t = np.clip(0.5 + 0.22 * tones[strip], 0, 1)[:, None, None]
    img = base_a[None, None] * (1 - t) + base_b[None, None] * t
    img = np.repeat(img, px, axis=1)
    # fine grain: thin streaks along the strip direction (x), periodic in x so the tile repeats
    k = rng.integers(1, 6, 48)
    ph = rng.uniform(0, 2 * np.pi, 48)
    amp = rng.uniform(0.2, 1.0, 48)
    yy = (y[:, None] / px)
    xx = (x[None, :] / px)
    grain = np.zeros((px, px))
    freq_y = rng.uniform(80, 400, 48)
    for i in range(48):
        grain += amp[i] * np.sin(2 * np.pi * freq_y[i] * yy + 3.0 * np.sin(2 * np.pi * k[i] * xx + ph[i]))
    grain /= np.abs(grain).max()
    streak = (grain > 0.55).astype(float) * (grain - 0.55) / 0.45
    img *= (1.0 - 0.10 * streak - 0.03 * grain)[..., None]
    # glue lines between strips
    edge = ((y * n_strip) % px) < max(1, px // 1024)
    img[edge] *= 0.86
    # a few darker mineral streaks
    for _ in range(6):
        r0 = rng.integers(0, px); w = rng.integers(2, 6); x0 = rng.integers(0, px); L = rng.integers(px // 8, px // 3)
        xs = (x0 + np.arange(L)) % px
        img[r0:r0 + w][:, xs] *= np.array([0.86, 0.82, 0.78])
    # strips run along the texture's v axis = the table's width (world y), like the real top
    img = np.transpose(img, (1, 0, 2))
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def make_table_texture():
    img = make_butcher_block()
    d = os.path.join(OUT, "table")
    os.makedirs(d, exist_ok=True)
    img.save(os.path.join(d, "maple.png"))
    os.makedirs(os.path.join(SIM, "web", "assets", "textures"), exist_ok=True)
    img.resize((1024, 1024), Image.LANCZOS).save(os.path.join(SIM, "web", "assets", "textures", "maple.jpg"), quality=90)
    print(f"[assets] table texture {img.size}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table-only", action="store_true", help="only the shared table texture (pick_cube preview)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SRC, exist_ok=True)
    if not args.table_only:
        build_bread("croissant")
        build_bread("bun")
        build_plate()
    make_table_texture()
