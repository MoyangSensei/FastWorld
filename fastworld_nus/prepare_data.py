"""Prepare the inputs for nuScenes validation."""

import argparse
from pathlib import Path

from .can import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_root
    streams = root / "can_bus"
    if (streams / "can_bus").is_dir():
        streams = streams / "can_bus"
    if not any(streams.glob("scene-*_pose.json")):
        parser.error(f"Extract the nuScenes supplemental data into {root}.")

    prepare(
        root / "vad_nuscenes_infos_temporal_train.pkl",
        root / "vad_nuscenes_infos_temporal_val.pkl",
        root / "v1.0-trainval" / "scene.json",
        streams,
        args.output,
    )


if __name__ == "__main__":
    main()
