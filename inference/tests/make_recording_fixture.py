"""Build an isolated test calibration and RGB pair from training recording metadata."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=100)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(args.params.read_text())
    params["policy"]["ckpt"] = str(
        (args.params.resolve().parent / params["policy"]["ckpt"]).resolve()
    )
    params["camera"].update(
        intrinsics_path="intrinsics.yaml", extrinsics_path="extrinsics.yaml"
    )
    params["perception"]["erase_prompt"] = "human arms . human hands ."
    params["deployment"]["name"] = "humanego-recording-validation"
    (folder / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    cam = source["cameras"]
    h, w = 480, 640
    original_w, original_h = cam["left"]["resolution"]
    scale = np.array(
        [
            [w / original_w, 0, (w / original_w - 1) / 2],
            [0, h / original_h, (h / original_h - 1) / 2],
            [0, 0, 1],
        ]
    )
    intrinsics = dict(
        image_size_hw=[h, w],
        rectified=True,
        **{
            eye: dict(K=(scale @ np.asarray(cam[eye]["k"])).tolist())
            for eye in ("left", "right")
        },
    )
    T = np.linalg.inv(cam["left"]["c2d"]) @ cam["right"]["c2d"]
    extrinsics = dict(
        translation_unit="m",
        T_left_from_right=T.tolist(),
        T_EE_from_hand=np.eye(4).tolist(),
    )
    (folder / "intrinsics.yaml").write_text(yaml.safe_dump(intrinsics))
    (folder / "extrinsics.yaml").write_text(yaml.safe_dump(extrinsics))
    cap = cv2.VideoCapture(source["source"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot decode requested SBS recording frame")
    pair = [
        cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
        for image in np.split(frame, 2, axis=1)
    ]
    state = np.array([0.15, 0.1, 0.6, 1, 0, 0, 1, 0, 0, 0], np.float32)
    np.savez_compressed(
        folder / "observation.npz", left=pair[0], right=pair[1], state=state
    )
    print(folder / "params.yaml")


if __name__ == "__main__":
    main()
