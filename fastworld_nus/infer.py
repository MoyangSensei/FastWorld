"""Run FastWorld on nuScenes validation and optionally compute LAW metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmcv import ConfigDict
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from . import dataset, model  # register the inference components
from .config import DATASET, MODEL

METRICS = tuple(f"plan_{name}_{second}s" for name in ("L2", "obj_box_col") for second in (1, 2, 3))


def load_planner(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu")
    if saved.get("format") != "fastworld_nus_inference_v1":
        raise ValueError("Expected the FastWorld nuScenes inference checkpoint.")
    planner = build_model(ConfigDict(MODEL))
    planner.load_state_dict(saved["state_dict"], strict=True)
    return planner.cuda().eval()


def summarize(rows):
    valid = [row["metrics"] for row in rows if row["metrics"]["fut_valid_flag"]]
    if not valid:
        raise ValueError("No predictions have valid 3-second ground truth.")
    l2 = [float(np.mean([row[key] for row in valid])) for key in METRICS[:3]]
    collision = [100 * float(np.mean([row[key] for row in valid])) for key in METRICS[3:]]
    return dict(samples=len(rows), valid=len(valid), seconds=[1, 2, 3], L2_m=l2, collision_percent=collision,
                average_L2_m=float(np.mean(l2)), average_collision_percent=float(np.mean(collision)))


@torch.inference_mode()
def run(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    validation = build_dataset(dict(
        DATASET, data_root=str(args.data_root.resolve()), ann_file=str(args.val_info.resolve()),
        can_features=str(args.can_features.resolve()),
    ))
    planner = load_planner(args.checkpoint)
    count = min(len(validation), args.limit) if args.limit else len(validation)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index in range(args.shard_index, count, args.shards):
            batch = scatter(collate([validation[index]], samples_per_gpu=1), [0])[0]
            images = batch["img"]
            metadata = batch["img_metas"]
            increments = planner.predict_trajectory(images, metadata)
            positions = increments.cumsum(dim=-2)
            token = metadata[0][images.shape[1] - 1]["sample_idx"]
            row = dict(sample_token=token, trajectory_increments=increments[0].tolist(),
                       trajectory=positions[0].tolist())
            if args.score:
                target = batch["ego_fut_trajs"][0][0, 0, :6].cumsum(dim=-2)
                row["metrics"] = planner.compute_planner_metric_stp3(
                    positions, target[None], batch["gt_bboxes_3d"][0][0],
                    batch["gt_attr_labels"][0][0].cpu().unsqueeze(0), bool(batch["fut_valid_flag"][0][0]),
                )
            output.write(json.dumps(row) + "\n")
            rows.append(row)
    if args.score and any(row["metrics"]["fut_valid_flag"] for row in rows):
        print(json.dumps(summarize(rows), indent=2))
    print(f"Wrote {len(rows)} predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data-root", "val-info", "can-features", "checkpoint", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--score", action="store_true", help="Compute LAW metrics using validation annotations.")
    parser.add_argument("--limit", type=int, help="Use the first N samples for a quick check.")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards or (args.limit is not None and args.limit < 1):
        parser.error("Invalid shard or sample limit.")
    run(args)


if __name__ == "__main__":
    main()
