"""Batch PICO SpatialMP4 adaptation without changing the Aria pipeline.

Run in the HumanEgo environment (including its optional HaMeR dependencies)::

    python -m preprocess.SpatialVideoPreprocess --input-dir /recordings --task TASK
    python -m preprocess.SpatialVideoPreprocess --input-dir /recordings --task TASK --stage downstream

Only direct MP4 children are scanned. Outputs are fixed at
``<repository>/data/<task>/spatial/spatial_<task>_000/preprocess/``. The adjacent
``spatial_index.json`` reserves stable numbers by source SHA256. Completed stages
are skipped; changed parameters and failed/incomplete runs require manual review,
never automatic overwriting. Failed outputs are retained in ``.spatial_failed``.

Requires ffmpeg/ffprobe and the original Python environment. No SpatialMP4 SDK or
Aria hardware is required, although importing the original pipeline still needs
projectaria-tools. The supported format is PICO's calibrated side-by-side RGB,
head pcrd=3/dtac=9, and optional raw1/dtmm depth. Right RGB and depth remain in the
untouched source file. HaMeR uses the existing monocular reconstruction, nominal
FPS, phase heuristics and object triangulation; this adapter does not improve
their accuracy or fuse stereo/depth. The default head offset is an unverified
official reference, not a measured device calibration.
LaMa GPU execution also requires ONNX Runtime matching the installed CUDA/cuDNN;
the original module can otherwise fall back to CPU.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERSION = 1
HEAD_OFFSET = (-0.05057, -0.01874, 0.04309)
HEAD_AXES = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
HEAD_REFERENCE = "https://github.com/Pico-Developer/SpatialMP4/blob/main/README.md"
FRAME_FILES = ("rgb.png", "aria_cam_rgb.json", "aria_hands.json",
               "hamer_hands.json", "aria_slam.json", "aria_phases.json")
DOWNSTREAM_STEPS = ("indices", "dinosam", "kptsselector", "cotracker",
                    "camtriangulator", "lama", "visualkpts", "datasetgen")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path, *arguments: str) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", *arguments, "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _boxes(data: bytes):
    position = 0
    while position < len(data):
        if position + 8 > len(data):
            raise ValueError("Truncated MP4 box header")
        size, kind = struct.unpack_from(">I4s", data, position)
        header_size = 8
        if size == 1:
            if position + 16 > len(data):
                raise ValueError("Truncated extended MP4 box header")
            size = struct.unpack_from(">Q", data, position + 8)[0]
            header_size = 16
        if size == 0:
            size = len(data) - position
        if size < header_size or position + size > len(data):
            raise ValueError("Invalid MP4 box size")
        yield kind, data[position + header_size:position + size]
        position += size


def _find_box(data: bytes, *kinds: bytes) -> bytes:
    for kind, payload in _boxes(data):
        if kind == kinds[0]:
            return _find_box(payload, *kinds[1:]) if len(kinds) > 1 else payload
    raise ValueError(f"Missing MP4 box {kinds[0]!r}")


def _decode_metadata(name: str, payload: bytes) -> Any:
    if name in ("icam", "ecam"):
        start, count = (18, 4) if name == "icam" else (8, 12)
        values = [math.ldexp(struct.unpack_from("<d", payload, start + i * 9 + 1)[0],
                             struct.unpack_from("b", payload, start + i * 9)[0])
                  for i in range(count)]
        return {"view_id": struct.unpack_from(">H", payload, 4)[0] & 1023, "values": values}
    if name == "tbtm":
        return struct.unpack(">Q", payload)[0]
    if name in ("pcrd", "dtac"):
        return struct.unpack_from(">I", payload, 4)[0]
    if name == "ppos":
        return payload[4:].rstrip(b"\0").decode()
    if name == "dstr":
        model_end = payload.index(b"\0", 4)
        camera_end = payload.index(b"\0", model_end + 1)
        values = payload[camera_end + 1:]
        return {"distortion_model": payload[4:model_end].decode(),
                "camera_model": payload[model_end + 1:camera_end].decode(),
                "values": list(struct.unpack("<" + "d" * (len(values) // 8), values))}
    return payload.hex()


def _sample_metadata(data: bytes) -> dict:
    # Sample-entry headers vary; spatial boxes carry their own bounded size.
    metadata = {}
    for name in ("icam", "ecam", "tbtm", "dstr", "pcrd", "dtac", "ppos", "rawC"):
        position = 0
        while True:
            position = data.find(name.encode(), position)
            if position < 0:
                break
            if position >= 4:
                size = struct.unpack_from(">I", data, position - 4)[0]
                if size >= 8 and position - 4 + size <= len(data):
                    value = _decode_metadata(name, data[position + 4:position - 4 + size])
                    if name in ("icam", "ecam"):
                        metadata.setdefault(name, []).append(value)
                    elif name in metadata:
                        raise ValueError(f"Multiple {name} boxes are not supported")
                    else:
                        metadata[name] = value
            position += 4
    return metadata


def read_spatial_metadata(path: Path) -> list[dict]:
    """Read calibration/sample entries without loading media payloads."""
    with path.open("rb") as stream:
        file_size = path.stat().st_size
        while stream.tell() < file_size:
            position = stream.tell()
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated top-level MP4 header")
            size, kind = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError("Truncated extended MP4 header")
                size = struct.unpack(">Q", extended)[0]
                header_size = 16
            if size == 0:
                size = file_size - position
            if size < header_size or position + size > file_size:
                raise ValueError("Invalid top-level MP4 size")
            if kind == b"moov":
                return [_sample_metadata(_find_box(track, b"mdia", b"minf", b"stbl", b"stsd"))
                        for box_kind, track in _boxes(stream.read(size - header_size))
                        if box_kind == b"trak"]
            stream.seek(position + size)
    raise ValueError("MP4 has no moov box")


def _timestamp_ns(pts: int, stream: dict, metadata: dict) -> int:
    if "tbtm" not in metadata:
        raise ValueError("Spatial track is missing tbtm epoch")
    return int(metadata["tbtm"]) * 1000 + round(
        Fraction(int(pts)) * Fraction(stream["time_base"]) * 1_000_000_000)


def _calibration(metadata: dict, view: int, size: tuple[int, int]) -> dict:
    intrinsics = [item["values"] for item in metadata.get("icam", []) if item["view_id"] == view]
    extrinsics = [item["values"] for item in metadata.get("ecam", []) if item["view_id"] == view]
    if len(intrinsics) != 1 or len(extrinsics) != 1:
        raise ValueError(f"Missing or ambiguous camera calibration for view {view}")
    fx, fy, cx, cy = intrinsics[0]
    transform = np.eye(4)
    transform[:3] = np.asarray(extrinsics[0]).reshape(3, 4)
    rotation = transform[:3, :3]
    if (not np.isfinite(transform).all() or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("Camera extrinsic is not a finite proper rigid transform")
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid camera intrinsics")
    distortion = metadata.get("dstr", {})
    if distortion and distortion.get("distortion_model") != "brown":
        raise ValueError("Only Brown pinhole distortion is supported")
    coefficients = distortion.get("values", [0.] * 5)
    if len(coefficients) not in (4, 5, 8, 12, 14) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid Brown distortion coefficients")
    return {"resolution": list(size), "k": [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
            "d": coefficients, "c2d": transform.tolist()}


@dataclass
class SpatialRecording:
    streams: list[dict]
    metadata: list[dict]
    rgb_index: int
    head_index: int
    rgb_timestamps_ns: np.ndarray
    head_timestamps_ns: np.ndarray
    head_samples: np.ndarray
    cameras: dict
    depth: dict | None


def read_recording(path: Path) -> SpatialRecording:
    """Read this PICO format's original timestamps, poses and calibration."""
    metadata = read_spatial_metadata(path)
    streams = _probe(path, "-show_streams")["streams"]
    if len(streams) != len(metadata):
        raise ValueError("MP4/ffprobe track order differs")
    rgb = [i for i, item in enumerate(metadata) if len(item.get("icam", [])) == 2
           and streams[i].get("codec_type") == "video"]
    head = [i for i, item in enumerate(metadata) if item.get("ppos") == "head"]
    if len(rgb) != 1 or len(head) != 1:
        raise ValueError("Expected one calibrated stereo RGB track and one head pose track")
    rgb_index, head_index = rgb[0], head[0]
    if metadata[head_index].get("pcrd") != 3 or metadata[head_index].get("dtac") != 9:
        raise ValueError("Unsupported PICO head format: expected pcrd=3/dtac=9")
    width, height = streams[rgb_index]["width"], streams[rgb_index]["height"]
    if width <= 0 or width % 2 or height <= 0:
        raise ValueError("Invalid side-by-side stereo dimensions")
    cameras = {side: _calibration(metadata[rgb_index], view, (width // 2, height))
               for view, side in enumerate(("left", "right"))}
    packets = _probe(path, "-show_packets", "-show_entries",
                     "packet=stream_index,pts,pos,size")["packets"]
    head_packets = [p for p in packets if p["stream_index"] == head_index]
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        samples = []
        for packet in head_packets:
            position = int(packet["pos"])
            if int(packet["size"]) != 56 or position < 0 or position + 56 > file_size:
                raise ValueError("Invalid head pose packet bounds or size")
            stream.seek(position)
            samples.append(struct.unpack("<7d", stream.read(56)))
    head_samples = np.asarray(samples, dtype=np.float64)
    head_times = np.array([_timestamp_ns(p["pts"], streams[head_index], metadata[head_index])
                           for p in head_packets], dtype=np.int64)
    if len(head_times) < 2 or not np.isfinite(head_samples).all() or np.any(np.diff(head_times) <= 0):
        raise ValueError("Head poses need finite samples and strictly increasing timestamps")
    if np.any(np.abs(np.linalg.norm(head_samples[:, 3:], axis=1) - 1.) > 1e-3):
        raise ValueError("Head quaternions must have unit norm (tolerance 0.001)")
    frames = _probe(path, "-select_streams", str(rgb_index), "-show_frames", "-show_entries",
                    "frame=best_effort_timestamp,width,height")["frames"]
    if any((f.get("width"), f.get("height")) != (width, height) for f in frames):
        raise ValueError("Variable RGB resolution is not supported")
    rgb_times = np.array([_timestamp_ns(f["best_effort_timestamp"], streams[rgb_index],
                                       metadata[rgb_index]) for f in frames], dtype=np.int64)
    if len(rgb_times) < 2 or np.any(np.diff(rgb_times) <= 0):
        raise ValueError("RGB timestamps must be strictly increasing")
    depth_indices = [i for i, stream in enumerate(streams) if stream.get("codec_tag_string") == "raw1"]
    if len(depth_indices) > 1:
        raise ValueError("Multiple depth tracks are not supported")
    depth = None
    if depth_indices:
        index = depth_indices[0]
        if not metadata[index].get("rawC", "").endswith(b"dtmm".hex()):
            raise ValueError("Unsupported raw1 depth format; expected dtmm")
        size = (streams[index]["width"], streams[index]["height"])
        calibration = _calibration(metadata[index], 0, size)
        depth_packets = [p for p in packets if p["stream_index"] == index]
        for packet in depth_packets:
            if (int(packet["size"]) != size[0] * size[1] * 2 or int(packet["pos"]) < 0 or
                    int(packet["pos"]) + int(packet["size"]) > file_size):
                raise ValueError("Invalid depth packet size or bounds")
        times = [_timestamp_ns(p["pts"], streams[index], metadata[index]) for p in depth_packets]
        if np.any(np.diff(np.asarray(times, dtype=np.int64)) <= 0):
            raise ValueError("Depth timestamps must increase strictly")
        depth = {"stream_index": index, "calibration": calibration, "frame_count": len(times),
                 "timestamps_ns": times, "raw_value_unit": "mm", "raw_dtype": "uint16_le",
                 "note": "Unfiltered raw depth remains in source; no RGB registration or fusion."}
    return SpatialRecording(streams, metadata, rgb_index, head_index, rgb_times,
                            head_times, head_samples, cameras, depth)


def camera_trajectory(recording: SpatialRecording, offset: Sequence[float]) -> dict:
    """Interpolate head samples, apply HeadToImu, then use a Z-up world."""
    first, last = recording.head_timestamps_ns[[0, -1]]
    selected = np.flatnonzero((recording.rgb_timestamps_ns >= first) &
                              (recording.rgb_timestamps_ns <= last))
    if len(selected) < 2:
        raise ValueError("Fewer than two RGB frames lie inside head pose coverage")
    times = recording.rgb_timestamps_ns[selected]
    # Subtract the integer epoch before converting to seconds for interpolation.
    head_seconds = (recording.head_timestamps_ns - first) / 1e9
    rgb_seconds = (times - first) / 1e9
    position = np.column_stack([np.interp(rgb_seconds, head_seconds, recording.head_samples[:, i])
                                for i in range(3)])
    rotation = Slerp(head_seconds, Rotation.from_quat(recording.head_samples[:, 3:]))(rgb_seconds).as_matrix()
    device = np.repeat(np.eye(4)[None], len(times), axis=0)
    device[:, :3, :3] = HEAD_AXES @ rotation @ HEAD_AXES.T
    device[:, :3, 3] = (position - np.einsum("nij,j->ni", rotation, offset)) @ HEAD_AXES.T
    c2d = np.asarray(recording.cameras["left"]["c2d"])
    camera0 = device[0] @ c2d
    up = HEAD_AXES @ np.array([0., 1., 0.])
    forward = camera0[:3, 2] - up * np.dot(up, camera0[:3, 2])
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("Initial camera heading is vertical; cannot define horizontal world X")
    forward /= np.linalg.norm(forward)
    source_to_world = np.eye(4)
    source_to_world[:3, :3] = np.stack((forward, np.cross(up, forward), up))
    source_to_world[:3, 3] = -source_to_world[:3, :3] @ camera0[:3, 3]
    d2w = source_to_world @ device
    intervals = np.diff(times) / 1e9
    nominal_period = float(np.median(np.diff(recording.rgb_timestamps_ns)) / 1e9)
    fps = int(math.floor(1. / nominal_period + 0.5))
    if fps < 1:
        raise ValueError("Invalid nominal RGB frame rate")
    return {"source_indices": selected, "timestamps_ns": times, "c2d": c2d,
            "d2w": d2w, "c2w": d2w @ c2d, "source_to_world": source_to_world,
            "fps": fps, "timing": {"nominal_fps": fps,
                "min_interval_ms": float(intervals.min() * 1000),
                "median_interval_ms": float(np.median(intervals) * 1000),
                "max_interval_ms": float(intervals.max() * 1000),
                "estimated_dropped_frames": int(np.maximum(np.rint(intervals / nominal_period) - 1, 0).sum()),
                "head_max_gap_ms": float(np.diff(recording.head_timestamps_ns).max() / 1e6)}}


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise ValueError(f"Decoded RGB ended early ({len(data)}/{size} bytes)")
        data.extend(chunk)
    return bytes(data)


def extract_images(source: Path, session: Path, recording: SpatialRecording, trajectory: dict) -> None:
    import cv2

    width, height = recording.cameras["left"]["resolution"]
    selected = {int(original): idx for idx, original in enumerate(trajectory["source_indices"])}
    calibration = recording.cameras["left"]
    k = np.asarray(calibration["k"])
    distortion = np.asarray(calibration["d"])
    maps = None
    if np.any(distortion != 0):
        maps = cv2.initUndistortRectifyMap(k, distortion, None, k, (width, height), cv2.CV_32FC1)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-map",
               f"0:{recording.rgb_index}", "-an", "-sn", "-dn", "-vf",
               f"crop={width}:{height}:0:0", "-vsync", "0", "-threads", "1",
               "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            for original in range(len(recording.rgb_timestamps_ns)):
                image = np.frombuffer(_read_exact(process.stdout, width * height * 3), dtype=np.uint8)
                if original not in selected:
                    continue
                image = image.reshape(height, width, 3)
                if maps is not None:
                    image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
                frame = session / "preprocess" / "all_data" / f"{selected[original]:05d}"
                frame.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(frame / "rgb.png"), image):
                    raise OSError(f"Could not write {frame / 'rgb.png'}")
            if process.stdout.read(1):
                raise ValueError("Decoded RGB has more frames than ffprobe timestamps")
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg decode failed: {errors.read().decode(errors='replace')}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


class DiskFrames(Sequence):
    """Supply real AriaCamData objects while keeping only one PNG in memory."""

    def __init__(self, frames: list, session: Path):
        self.frames = frames
        self.session = session

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):
        import cv2

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        frame = self.frames[index]
        path = self.session / "preprocess" / "all_data" / f"{frame.idx:05d}" / "rgb.png"
        image = cv2.imread(str(path))
        if image is None or image.shape != (frame.h, frame.w, 3):
            raise ValueError(f"Missing or invalid RGB image: {path}")
        return replace(frame, img=image)


def build_camera(session: Path, recording: SpatialRecording, trajectory: dict):
    from preprocess.AriaCamTypes import AriaCam, AriaCamData

    calibration = recording.cameras["left"]
    width, height = calibration["resolution"]
    k = np.asarray(calibration["k"])
    camera = AriaCam(mps_path=str(session))
    camera.tss = trajectory["timestamps_ns"].tolist()
    camera.fps = trajectory["fps"]
    camera.first_ts = camera.tss[0]
    camera.w, camera.h = width, height
    camera.k, camera.d, camera.c2d = k, np.zeros(5), trajectory["c2d"]
    camera.fov = math.degrees(2 * math.atan(height / (2 * k[1, 1])))
    frames = [AriaCamData(idx=i, ts=ts, w=width, h=height, k=k, d=camera.d,
                         fov=camera.fov, c2d=camera.c2d, d2w=trajectory["d2w"][i],
                         c2w=trajectory["c2w"][i]) for i, ts in enumerate(camera.tss)]
    camera.cam = DiskFrames(frames, session)
    return camera


def validate_prepared(session: Path, settings: dict, source_hash: str, *, require_complete: bool = True) -> dict:
    record = _read_json(session / "spatial_source.json")
    if ((require_complete and not record.get("prepare_complete")) or record.get("source_sha256") != source_hash or
            record.get("settings") != settings):
        raise ValueError("Prepared source/settings do not match this run")
    config = _read_json(session / "preprocess" / "aria_cam_rgb_config.json")
    count = config["total_frames"]
    if count != len(record["frame_mapping"]) or count < 2:
        raise ValueError("Prepared frame count differs from source mapping")
    frame_root = session / "preprocess" / "all_data"
    names = sorted(p.name for p in frame_root.iterdir() if p.is_dir() and p.name.isdigit())
    if names != [f"{i:05d}" for i in range(count)]:
        raise ValueError("Prepared frame directories are not contiguous")
    previous_ts = None
    valid_hands = 0
    for i, mapping in enumerate(record["frame_mapping"]):
        frame = frame_root / f"{i:05d}"
        if any(not (frame / name).is_file() or (frame / name).stat().st_size == 0 for name in FRAME_FILES):
            raise ValueError(f"Incomplete frame: {frame}")
        camera = _read_json(frame / "aria_cam_rgb.json")
        hands = _read_json(frame / "aria_hands.json")
        if hands != _read_json(frame / "hamer_hands.json"):
            raise ValueError(f"Canonical/HaMeR hand results differ at frame {i}")
        ts = camera["ts"]
        if (camera["idx"] != i or hands["idx"] != i or hands["ts"] != ts or
                mapping["idx"] != i or mapping["timestamp_ns"] != ts or
                (previous_ts is not None and ts <= previous_ts)):
            raise ValueError(f"Inconsistent frame identity/timestamp at {i}")
        c2w = np.asarray(camera["c2w"])
        if (not np.isfinite(c2w).all() or not np.allclose(
                c2w, np.asarray(camera["d2w"]) @ np.asarray(camera["c2d"]), atol=1e-7)):
            raise ValueError(f"Inconsistent camera transform at frame {i}")
        for name in ("aria_slam.json", "aria_phases.json"):
            data = _read_json(frame / name)
            if data["idx"] != i or data["ts"] != ts:
                raise ValueError(f"Inconsistent {name} at frame {i}")
        for side in ("hand_r", "hand_l"):
            hand = hands.get(side)
            if hand and hand.get("midpoint_pose_opt_world") is not None:
                pose = np.asarray(hand["midpoint_pose_opt_world"])
                if not np.isfinite(pose).all():
                    raise ValueError(f"Non-finite hand pose at frame {i}")
                valid_hands += 1
        previous_ts = ts
    if not valid_hands:
        raise ValueError("No optimized hand poses remain; inspect tracking and task hand selection")
    if not (session / "preprocess" / "aria_phases_results.json").is_file():
        raise ValueError("Missing phase summary")
    return record


def validate_training(session: Path) -> dict:
    paths = list((session / "preprocess" / "all_data").glob("*/training_data.json"))
    if not paths:
        raise ValueError("No training_data.json generated; check phases, hands and object prompts")
    usable = 0
    for path in paths:
        data = _read_json(path)
        for name in ("rgb_path", "rgb_WoArm_path", "rgb_WoArm_WArmObjKpts_path",
                     "mask_arm_path", "mask_obj_path"):
            image = Path(data["obs"][name])
            if not image.is_file() or image.stat().st_size == 0:
                raise ValueError(f"Training image/mask missing: {image}")
        if data.get("entities", {}).get("hands") and data.get("entities", {}).get("objects"):
            usable += 1
    if not usable:
        raise ValueError("Training JSONs contain no usable hand/object samples")
    return {"training_frames": len(paths), "usable_training_frames": usable}


def _export_hands_video(camera, hands, path: Path, hand_cfg) -> None:
    import cv2
    from preprocess.AriaHandsOps import AriaHandsOps

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), camera.fps, (camera.w, camera.h))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer: {path}")
    try:
        for frame, data in zip(camera.cam, hands.hands):
            image = AriaHandsOps.draw_aria_hands_skeleton(
                frame.img, data, frame.k, frame.d, frame.c2w, hand_cfg.grasp_threshold)
            writer.write(image)
    finally:
        writer.release()


def generate_phases(generator):
    """Use the original phase operations with an explicit 1D mask boundary.

    OpenCV 5 returns a row image for a 1D morphology input. The original
    generator passes that image to 1D run-length operations unchanged.
    Keeping the orchestration here avoids patching its shared Ops class.
    """
    from preprocess.AriaPhasesOps import AriaPhasesOps as ops
    from preprocess.AriaPhasesTypes import AriaPhases, AriaPhasesFrame

    cfg = generator.cfg
    frames = generator.aria_slam.frames
    v = np.array([frame.v for frame in frames])
    w = np.array([frame.w for frame in frames])
    yaw = np.array([frame.yaw_unwrapped_deg for frame in frames])
    stop = ops.compute_stop_mask_from_vw(v, w, cfg.v_stop_thresh, cfg.w_stop_thresh,
                                         cfg.stop_hold_frames, cfg.stop_debounce_frames)
    # Explicit column input also keeps the vertical kernel on the time axis.
    stop = np.asarray(ops.close_binary_mask(stop.reshape(-1, 1), cfg.stop_debounce_frames)).reshape(-1)
    stop = ops.postprocess_stop_mask(stop, yaw, cfg.stop_min_on_frames, cfg.stop_yaw_veto_deg)
    stop = ops.apply_stop_offset(stop, cfg.stop_offset_frames)
    mode, _, stats = ops.compute_mode_from_stop_vw(stop, v, w, yaw, cfg.w_rot_thresh, cfg.v_rot_max, cfg)
    if generator.aria_hands is not None:
        mode, stats = ops.refine_manip_phases_with_hand_vel(
            mode, generator.aria_hands, getattr(cfg, "manip_clean_vel_thresh", 0.15),
            getattr(cfg, "manip_clean_wait_frames", 5), getattr(cfg, "manip_clean_manual_offset", 15))
    mode = ops.inject_finished_phase(mode, getattr(cfg, "finished_frames", 60))
    names = {0: "STOP", 1: "FORWARD", 2: "ROTATE", 3: "TRANSITION", 4: "FINISHED"}
    phases = AriaPhases(mps_path=generator.mps_path)
    phases.frames = [AriaPhasesFrame(idx=frame.idx, ts=frame.ts, mode=int(mode[i]), stop=int(stop[i]),
                                     mode_str=names[int(mode[i])], v=float(v[i]), w=float(w[i]),
                                     yaw_u_deg=float(yaw[i])) for i, frame in enumerate(frames)]
    phases.summary = {
        "total_frames": len(frames), "fps_median": generator.aria_cam.fps,
        "duration_s": (frames[-1].ts - frames[0].ts) / 1e9,
        "segment_summary_opt": ops.segment_summary_from_masks(stop, v, w, cfg.v_walk_thresh),
        "stage_window_check": ops.stage_window_check(mode, [4, 3, 2, 1, 0], True, False),
        "mode_stats_opt": stats, "hyperparams": cfg.to_dict(),
    }
    return phases


def release_stage_models(step: str) -> None:
    """Later stages consume files, so their predecessors need not retain VRAM."""
    import gc
    import torch

    if step == "dinosam":
        from preprocess import DINOSAMOps
        if DINOSAMOps._WORKER is not None:
            DINOSAMOps._WORKER.close()
            DINOSAMOps._WORKER = None
    elif step == "cotracker":
        from preprocess.CoTrackerOffline import reset_cotracker_offline
        reset_cotracker_offline()
    else:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pipeline_class():
    # Heavy imports stay in workers, including when only --help is requested.
    from preprocess.Preprocess import Preprocess
    from preprocess.AriaHands import AriaHandsGenerator
    from preprocess.AriaSlam import AriaSlamGenerator
    from preprocess.AriaPhases import AriaPhasesGenerator
    from preprocess.HaMeRHands import HaMeRHandsGenerator

    class SpatialVideoPreprocess(Preprocess):
        def __init__(self, job: dict):
            self.job = job
            super().__init__(mps_path=job["session"], cfg_path=job["cfg_path"], task=job["task"],
                             export_video=job["export_video"], export_gif=False)

        def init_preprocess(self) -> None:
            self.recording = read_recording(Path(self.job["source"]))
            self.trajectory = camera_trajectory(self.recording, self.job["settings"]["head_model_offset_m"])
            self.num_total_frames = len(self.trajectory["timestamps_ns"])
            self.start_idx, self.end_idx = 0, self.num_total_frames - 1
            self.cfg.AriaCam.fps = self.trajectory["fps"]

        def preprocess_aria(self) -> None:
            session = Path(self.mps_path)
            source = Path(self.job["source"])
            if _sha256(source) != self.job["source_sha256"]:
                raise ValueError("Source MP4 changed after batch registration")
            extract_images(source, session, self.recording, self.trajectory)
            camera = build_camera(session, self.recording, self.trajectory)
            camera.save_aria_cam_json(label="rgb")
            generator = HaMeRHandsGenerator(self.mps_path, self.cfg.AriaHands_path, camera)
            if not generator.hamer_model.is_available:
                raise RuntimeError("HaMeR model unavailable; refusing MediaPipe-only 3D fallback")
            hands = generator.get_aria_hands()
            # This selection method uses only cfg and the supplied hands.
            AriaHandsGenerator._apply_hand_selection(generator, hands)
            hands.save_hands_json(filename="hamer_hands.json")
            hands.save_hands_json(filename="aria_hands.json")
            detector = generator._detector_name
            del generator
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            slam = AriaSlamGenerator(self.mps_path, self.cfg.AriaSlam_path, camera).get_aria_slam()
            slam.save_aria_slam_json()
            phases = generate_phases(AriaPhasesGenerator(
                self.mps_path, self.cfg.AriaPhases_path, camera, slam, hands))
            phases.save_aria_phases_json()
            if self.export_video:
                _export_hands_video(camera, hands, session / "preprocess" / "vis" / "hamer_hands_vis.mp4",
                                    self.cfg.AriaHands)
            if _sha256(source) != self.job["source_sha256"]:
                raise ValueError("Source MP4 changed during preparation")
            record = {"version": VERSION, "source": str(source), "source_sha256": self.job["source_sha256"],
                      "sensor_source": "pico_spatial_mp4", "hand_source": "hamer", "hand_detector": detector,
                      "settings": self.job["settings"], "prepare_complete": True, "downstream_complete": False,
                      "head_model": {"offset_m": self.job["settings"]["head_model_offset_m"],
                                     "source": self.job["settings"]["head_model_source"], "reference": HEAD_REFERENCE},
                      "camera_frame": "opencv_x_right_y_down_z_forward",
                      "world_frame": "right_handed_z_up_x_initial_heading_y_left",
                      "source_to_world": self.trajectory["source_to_world"].tolist(),
                      "streams": self.recording.streams, "spatial_metadata": self.recording.metadata,
                      "cameras": self.recording.cameras, "depth": self.recording.depth,
                      "rgb_frame_count": len(self.recording.rgb_timestamps_ns),
                      "head_pose_samples": len(self.recording.head_samples), "timing": self.trajectory["timing"],
                      "frame_mapping": [{"idx": i, "source_frame_idx": int(original), "timestamp_ns": int(ts)}
                                        for i, (original, ts) in enumerate(zip(self.trajectory["source_indices"], camera.tss))]}
            # Validate before publishing the completion marker.
            record["prepare_complete"] = False
            _write_json(session / "spatial_source.json", record)
            validate_prepared(session, self.job["settings"], self.job["source_sha256"], require_complete=False)
            record["prepare_complete"] = True
            _write_json(session / "spatial_source.json", record)

        def run(self) -> None:
            session = Path(self.mps_path)
            source_record = session / "spatial_source.json"
            if source_record.is_file():
                record = validate_prepared(session, self.job["settings"], self.job["source_sha256"])
                summary = _read_json(session / "preprocess" / "aria_cam_rgb_config.json")
                self.num_total_frames = summary["total_frames"]
                self.cfg.AriaCam.fps = summary["fps"]
            else:
                if self.job["stage"] == "downstream":
                    raise ValueError("downstream requires completed prepare outputs")
                self.init_preprocess()
                self.preprocess_aria()
                record = _read_json(source_record)
            if self.job["stage"] == "prepare":
                return
            if record.get("downstream_complete"):
                validate_training(session)
                return
            for step in DOWNSTREAM_STEPS:
                print(f"[Spatial] {session.name}: {step}", flush=True)
                getattr(self, f"preprocess_{step}")()
                release_stage_models(step)
            record.update(validate_training(session))
            record["downstream_complete"] = True
            _write_json(source_record, record)

    return SpatialVideoPreprocess


def _load_settings(cfg_path: Path, task: str, offset, export_video: bool) -> dict:
    import yaml
    from utils.utils_io import deep_update

    task_path = cfg_path.parent.parent / "tasks" / f"{task}.yaml"
    if not cfg_path.is_file() or not task_path.is_file():
        raise FileNotFoundError(f"Base and task YAML must exist: {cfg_path}, {task_path}")
    dependencies = {}
    configs = {}
    visiting = set()

    def visit(path: Path) -> dict:
        path = path.resolve()
        if str(path) in configs:
            return configs[str(path)]
        if path in visiting:
            raise ValueError(f"Circular configuration reference: {path}")
        visiting.add(path)
        dependencies[str(path)] = _sha256(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        compiled = data.copy()
        for key, value in data.items():
            # The released master references a missing, unused Aria MPS config.
            if key.endswith("_path") and key != "AriaMPS_path" and isinstance(value, str):
                child = Path(value)
                if not child.is_file():
                    child = path.parent / value
                if not child.is_file():
                    raise FileNotFoundError(f"Missing configuration {key}: {value}")
                compiled[key[:-5]] = visit(child)
        visiting.remove(path)
        configs[str(path)] = compiled
        return compiled

    config = visit(cfg_path)
    visit(task_path)
    config = deep_update(config, yaml.safe_load(task_path.read_text(encoding="utf-8")) or {})
    selection = config.get("AriaHands", {}).get("hand_selection", "none")
    if selection not in ("none", "left", "right", "auto"):
        raise ValueError(f"Invalid hand_selection: {selection}")
    if not np.isfinite(offset).all():
        raise ValueError("Head model offset must contain three finite values")
    return {"version": VERSION, "task": task, "config_sha256": dependencies,
            "head_model_offset_m": list(offset),
            "head_model_source": "official_reference_unverified" if tuple(offset) == HEAD_OFFSET else "explicit_cli_unverified",
            "primary_eye": "left", "hand_source": "hamer", "export_video": export_video,
            "object_prompts": config.get("DINOSAM", {}).get("dinosam_prompt", {})}


def _check_prompts(settings: dict) -> None:
    prompts = settings["object_prompts"]
    objects = [value for key, value in prompts.items() if key.startswith("obj")]
    if not objects or any(not isinstance(p, str) or not p.strip() or "pretty cat" in p.lower() for p in objects):
        raise ValueError("Task must define real DINOSAM object prompts, not the base placeholder")
    if not isinstance(prompts.get("arm"), str) or not prompts["arm"].strip():
        raise ValueError("Task must define DINOSAM arm prompt")


def _worker(job: dict, connection) -> None:
    os.setsid()
    os.chdir(ROOT)
    try:
        pipeline = _pipeline_class()
        if job.get("preflight"):
            if job["needs_prepare"]:
                from preprocess.HaMeRHands import HaMeRModel, ViTPoseHandDetector, MediaPipeHandDetector
                model = HaMeRModel()
                if not model.is_available:
                    raise RuntimeError("HaMeR weights/dependencies unavailable; activate the full HumanEgo hand environment")
                try:
                    ViTPoseHandDetector(device="cuda")
                except Exception as error:
                    print(f"[Spatial preflight] ViTPose unavailable ({error}); checking MediaPipe", flush=True)
                    MediaPipeHandDetector()
            connection.send({"ok": True})
            return
        current = _load_settings(Path(job["cfg_path"]), job["task"],
                                 job["settings"]["head_model_offset_m"], job["export_video"])
        if current != job["settings"] or _sha256(Path(job["source"])) != job["source_sha256"]:
            raise ValueError("Source or configuration changed after batch registration")
        pipeline(job).run()
        if _sha256(Path(job["source"])) != job["source_sha256"]:
            raise ValueError("Source changed while processing")
        connection.send({"ok": True})
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        connection.send({"ok": False, "error": error})
    finally:
        for module_name, attribute in (("preprocess.DINOSAMOps", "_WORKER"),
                                        ("preprocess.Lama", "_LAMA_WORKER")):
            module = sys.modules.get(module_name)
            worker = getattr(module, attribute, None) if module else None
            if worker is not None:
                worker.close()
        connection.close()


def _run_worker(job: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(job, sender))
    process.start()
    sender.close()
    try:
        try:
            result = receiver.recv()
        except EOFError:
            result = {"ok": False, "error": "Worker exited without a completion result"}
        process.join()
        if process.exitcode:
            return {"ok": False, "error": f"Worker exit code {process.exitcode}: {result.get('error', '')}"}
        return result
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.terminate()
        process.join()
        raise
    finally:
        # A killed/OOM worker cannot close its DINO/LaMa grandchildren itself.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        receiver.close()


def _validate_component(task: str) -> None:
    if not task or task in (".", "..") or "/" in task or "\\" in task or "\0" in task:
        raise ValueError("task must be one non-empty directory name")


def _load_index(task_dir: Path, task: str) -> dict:
    path = task_dir / "spatial_index.json"
    index = _read_json(path) if path.is_file() else {"version": VERSION, "task": task, "recordings": []}
    if index.get("version") != VERSION or index.get("task") != task:
        raise ValueError("Unsupported or mismatched spatial_index.json")
    entries = index["recordings"]
    numbers = [item["number"] for item in entries]
    hashes = [item["source_sha256"] for item in entries]
    if len(set(numbers)) != len(numbers) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate recording identities in spatial_index.json")
    for entry in entries:
        number = entry["number"]
        if not isinstance(number, int) or not 0 <= number <= 999:
            raise ValueError("Invalid recording number in index")
        expected = task_dir / "spatial" / f"spatial_{task}_{number:03d}"
        if entry["session"] != str(expected):
            raise ValueError("Index session path differs from fixed task layout")
    return index


def _next_number(task_dir: Path, task: str, index: dict) -> int:
    numbers = [entry["number"] for entry in index["recordings"]]
    pattern = re.compile(rf"spatial_{re.escape(task)}_(\d{{3}})$")
    for root in (task_dir / "spatial", task_dir / ".spatial_failed"):
        if root.is_dir():
            numbers.extend(int(match.group(1)) for path in root.iterdir()
                           if path.is_dir() and (match := pattern.fullmatch(path.name)))
    return max(numbers, default=-1) + 1


def _quarantine(entry: dict, task_dir: Path) -> None:
    session = Path(entry["session"])
    if not session.exists():
        return
    destination = task_dir / ".spatial_failed" / session.name
    if destination.exists():
        raise FileExistsError(f"Refusing to replace failed output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    session.rename(destination)
    entry["failed_output"] = str(destination)


def process_batch(args: argparse.Namespace) -> int:
    """Reserve stable identities, then process each source in an isolated worker."""
    _validate_component(args.task)
    input_dir = Path(args.input_dir).expanduser().resolve()
    cfg_path = Path(args.cfg_path).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    sources = sorted((p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"),
                     key=lambda p: p.name)
    if not sources:
        raise ValueError(f"No direct MP4 files in {input_dir}")
    os.chdir(ROOT)
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable missing: {executable}")
    settings = _load_settings(cfg_path, args.task, args.head_model_offset, args.export_video)
    if args.stage != "prepare":
        _check_prompts(settings)
    task_dir = ROOT / "data" / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / ".spatial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another spatial batch is using task {args.task}") from error
        return _process_locked_batch(args, sources, cfg_path, settings, task_dir)


def _process_locked_batch(args, sources: list[Path], cfg_path: Path, settings: dict, task_dir: Path) -> int:
    index = _load_index(task_dir, args.task)
    index_path = task_dir / "spatial_index.json"
    by_hash = {item["source_sha256"]: item for item in index["recordings"]}
    pending = []
    counts = {"success": 0, "skipped": 0, "failed": 0}
    failures = []

    def report_failure(source: Path, error: str) -> None:
        failures.append({"source": str(source), "error": error})
        counts["failed"] += 1
        print(f"[Spatial FAILED] {source.name}: {error}", flush=True)

    for source in sources:
        try:
            source_hash = _sha256(source)
        except OSError as error:
            report_failure(source, str(error))
            continue
        entry = by_hash.get(source_hash)
        if entry is None:
            if args.stage == "downstream":
                report_failure(source, "no prepared recording")
                continue
            number = _next_number(task_dir, args.task, index)
            if number > 999:
                raise ValueError("Spatial sequence capacity exceeded (000–999)")
            session = task_dir / "spatial" / f"spatial_{args.task}_{number:03d}"
            entry = {"number": number, "source": str(source), "source_sha256": source_hash,
                     "session": str(session), "settings": settings, "status": "reserved",
                     "prepare_complete": False, "downstream_complete": False, "error": None}
            index["recordings"].append(entry)
            by_hash[source_hash] = entry
            _write_json(index_path, index)
        if entry["settings"] != settings:
            report_failure(source, f"settings conflict with {entry['session']}")
            continue
        if entry["status"] in ("failed", "running"):
            if entry["status"] == "running":
                entry["status"] = "failed"
                entry["error"] = "Previous worker was interrupted; manual review required"
                _quarantine(entry, task_dir)
                _write_json(index_path, index)
            report_failure(source, "earlier run requires manual review")
            continue
        pending.append((source, entry))
    # A duplicate file in the same input directory must not enqueue its session twice.
    unique = {}
    for source, entry in pending:
        if entry["source_sha256"] in unique:
            counts["skipped"] += 1
        else:
            unique[entry["source_sha256"]] = (source, entry)
    pending = list(unique.values())
    jobs = []
    for source, entry in pending:
        completed = entry["prepare_complete"] if args.stage == "prepare" else entry["downstream_complete"]
        if completed:
            try:
                record = validate_prepared(Path(entry["session"]), settings, entry["source_sha256"])
                if args.stage != "prepare":
                    stats = validate_training(Path(entry["session"]))
                    if not record.get("downstream_complete") or any(record.get(k) != v for k, v in stats.items()):
                        raise ValueError("Completed recording has incomplete downstream results")
                print(f"[Spatial SKIP] {source.name} -> {Path(entry['session']).name}", flush=True)
                counts["skipped"] += 1
            except Exception as error:
                report_failure(source, str(error))
                entry["status"], entry["error"] = "failed", str(error)
                _quarantine(entry, task_dir)
                _write_json(index_path, index)
            continue
        if args.stage == "downstream" and not entry["prepare_complete"]:
            report_failure(source, "prepare is incomplete")
            continue
        jobs.append((source, entry))
    if jobs:
        result = _run_worker({"preflight": True, "needs_prepare": any(not e["prepare_complete"] for _, e in jobs)})
        if not result["ok"]:
            index["last_batch"] = {"input_dir": str(sources[0].parent), "stage": args.stage,
                                   **counts, "preflight_error": result["error"]}
            _write_json(index_path, index)
            raise RuntimeError(f"Global pipeline preflight failed:\n{result['error']}")
    for position, (source, entry) in enumerate(jobs, 1):
        print(f"[Spatial {position}/{len(jobs)}] {source.name} -> {Path(entry['session']).name}", flush=True)
        session = Path(entry["session"])
        if not entry["prepare_complete"] and session.exists():
            entry["status"] = "failed"
            entry["error"] = f"Refusing to overwrite incomplete/unregistered outputs: {session}"
            report_failure(source, entry["error"])
            _write_json(index_path, index)
            continue
        entry["status"], entry["error"] = "running", None
        _write_json(index_path, index)
        job = {"source": str(source), "source_sha256": entry["source_sha256"], "session": str(session),
               "settings": settings, "task": args.task, "cfg_path": str(cfg_path),
               "stage": args.stage, "export_video": args.export_video}
        try:
            result = _run_worker(job)
        except KeyboardInterrupt:
            entry["status"], entry["error"] = "failed", "Interrupted by user"
            _quarantine(entry, task_dir)
            _write_json(index_path, index)
            raise
        if result["ok"]:
            record = _read_json(session / "spatial_source.json")
            entry.update(status="complete" if record["downstream_complete"] else "prepared",
                         prepare_complete=record["prepare_complete"], downstream_complete=record["downstream_complete"])
            counts["success"] += 1
        else:
            entry["status"], entry["error"] = "failed", result["error"]
            _quarantine(entry, task_dir)
            report_failure(source, result["error"])
        _write_json(index_path, index)
    index["last_batch"] = {"input_dir": str(sources[0].parent), "stage": args.stage, **counts,
                           "errors": failures}
    _write_json(index_path, index)
    print(f"[Spatial batch] success={counts['success']} skipped={counts['skipped']} failed={counts['failed']}", flush=True)
    return 1 if counts["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory containing direct PICO SpatialMP4 files")
    parser.add_argument("--task", required=True, help="Task name; requires a matching preprocessing task YAML")
    parser.add_argument("--stage", choices=("prepare", "downstream", "all"), default="prepare")
    parser.add_argument("--cfg_path", default=str(ROOT / "cfg" / "preprocess" / "base" / "Preprocess.yaml"))
    parser.add_argument("--head-model-offset", type=float, nargs=3, default=HEAD_OFFSET, metavar=("X", "Y", "Z"))
    parser.add_argument("--export-video", action="store_true", help="Export videos (GIF is disabled)")
    args = parser.parse_args()
    try:
        return process_batch(args)
    except KeyboardInterrupt:
        print("[Spatial] Batch interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[Spatial] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
