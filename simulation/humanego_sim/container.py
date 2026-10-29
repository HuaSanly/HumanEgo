"""Rounded, tapered plastic container with a hollow visual mesh and wall contacts."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def rounded_outline(half: np.ndarray, radius: float, steps: int = 8) -> np.ndarray:
    points = []
    for sx, sy, angle in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        center = np.array([sx * (half[0] - radius), sy * (half[1] - radius)])
        for a in np.deg2rad(np.linspace(angle, angle + 90, steps, endpoint=False)):
            points.append(center + radius * np.array([np.cos(a), np.sin(a)]))
    return np.asarray(points)


def contains_points(points: np.ndarray, obj: dict, tolerance: float = 0.001) -> bool:
    """Test all cube corners against the tapered, rounded inner cavity."""
    half = np.asarray(obj["size"]) / 2
    wall = float(obj["wall"])
    if points[:, 2].min() < -half[2] + wall - 0.003 or points[:, 2].max() > half[2] + 0.003:
        return False
    return contains_xy(points, obj, tolerance)


def contains_xy(points: np.ndarray, obj: dict, tolerance: float = 0.001) -> bool:
    """Check horizontal clearance at each height; negative tolerance adds a margin."""
    half = np.asarray(obj["size"]) / 2
    wall = float(obj["wall"])
    if obj.get("shape") != "rounded":
        return bool(np.all(np.abs(points[:, :2]) <= half[:2] - wall + tolerance))
    taper = float(obj.get("taper", 0.84))
    t = np.clip((points[:, 2] + half[2] - wall) / (2 * half[2] - wall), 0, 1)
    scale = taper + (1 - taper) * t
    radius = np.maximum(float(obj.get("radius", 0.018)) * scale - wall, wall)
    half_xy = half[None, :2] * scale[:, None] - wall
    q = np.abs(points[:, :2]) - (half_xy - radius[:, None])
    distance = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - radius
    return bool(np.all(distance <= tolerance))


def add_rounded_container(spec, body, obj: dict) -> None:
    from .scene import G_COLLISION, G_OBJECT

    half = np.asarray(obj["size"], float) / 2
    wall = float(obj["wall"])
    radius = float(obj.get("radius", 0.018))
    taper = float(obj.get("taper", 0.84))
    rim = float(obj.get("rim", 0.005))
    top = rounded_outline(half[:2], radius)
    bottom = rounded_outline(half[:2] * taper, radius * taper)
    inner_top = rounded_outline(half[:2] - wall, max(radius - wall, wall))
    inner_bottom = rounded_outline(half[:2] * taper - wall, max(radius * taper - wall, wall))
    lip = rounded_outline(half[:2] + rim, radius + rim)
    rings = [(bottom, -half[2]), (top, half[2] - 0.004), (lip, half[2]),
             (inner_top, half[2]), (inner_bottom, -half[2] + wall)]
    vertices = np.concatenate([np.column_stack([p, np.full(len(p), z)]) for p, z in rings])
    faces = []
    n = len(top)
    for r in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = r * n + i, r * n + j, (r + 1) * n + j, (r + 1) * n + i
            faces.extend([[a, b, c], [a, c, d]])
    for i in range(1, n - 1):
        faces.extend([[0, i + 1, i], [4 * n, 4 * n + i, 4 * n + i + 1]])
    mesh = spec.add_mesh()
    mesh.name = f"{body.name}_shell"
    mesh.uservert = vertices.ravel().tolist()
    mesh.userface = np.asarray(faces).ravel().tolist()
    material = spec.add_material()
    material.name = f"{body.name}_plastic"
    material.rgba = obj["rgba"]
    material.specular = float(obj.get("specular", 0.15))
    material.shininess = float(obj.get("shininess", 0.25))
    visual = body.add_geom()
    visual.name = f"{body.name}_visual"
    visual.type = mujoco.mjtGeom.mjGEOM_MESH
    visual.meshname = mesh.name
    visual.material = material.name
    visual.group = G_OBJECT
    visual.contype = visual.conaffinity = 0
    visual.mass = 0

    contacts = []
    for i in range(n):
        j = (i + 1) % n
        lower = (bottom[i] + bottom[j]) / 2
        upper = (top[i] + top[j]) / 2
        tangent = top[j] - top[i]
        length = np.linalg.norm(tangent)
        tangent /= length
        slope = np.r_[upper - lower, 2 * half[2] - wall]
        slope /= np.linalg.norm(slope)
        normal = np.cross(np.r_[tangent, 0], slope)
        normal /= np.linalg.norm(normal)
        vertical = np.cross(np.r_[tangent, 0], normal)
        geom = body.add_geom()
        geom.name = f"{body.name}_wall_{i}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [length / 2 + 0.0004, wall / 2, np.linalg.norm(np.r_[upper - lower, 2 * half[2] - wall]) / 2]
        geom.pos = [*((upper + lower) / 2 - normal[:2] * wall / 2), wall / 2]
        rotation = np.column_stack([np.r_[tangent, 0], normal, vertical])
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        geom.quat = [w, x, y, z]
        contacts.append(geom)
    floor = body.add_geom()
    floor.name = f"{body.name}_bottom"
    floor.type = mujoco.mjtGeom.mjGEOM_BOX
    floor.size = [*(half[:2] * taper - wall), wall / 2]
    floor.pos = [0, 0, -half[2] + wall / 2]
    contacts.append(floor)
    for geom in contacts:
        geom.group = G_COLLISION
        geom.rgba = [0, 0, 0, 0]
        geom.mass = float(obj["mass"]) / len(contacts)
        geom.friction = obj["friction"]
    for sign in (-1, 1):
        handle = body.add_geom()
        handle.name = f"{body.name}_handle_{sign}"
        handle.type = mujoco.mjtGeom.mjGEOM_BOX
        handle.size = [0.025, 0.009, 0.003]
        handle.pos = [0, sign * (half[1] + 0.007), half[2] - 0.012]
        handle.material = material.name
        handle.group = G_OBJECT
        handle.contype = handle.conaffinity = 0
        handle.mass = 0
