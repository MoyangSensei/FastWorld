"""Two-frame validation loader backed by the LAW/VAD data format."""

from pathlib import Path

import numpy as np
from mmdet.datasets import DATASETS
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import VADCustomNuScenesDataset


@DATASETS.register_module()
class FastWorldNuScenesValidation(VADCustomNuScenesDataset):
    def __init__(self, can_features, **kwargs):
        super().__init__(**kwargs)
        with np.load(can_features, allow_pickle=False) as data:
            index = {token: i for i, token in enumerate(data["tokens"])}
            if len(index) != len(data["tokens"]):
                raise ValueError("Duplicate validation tokens in CAN features.")
            tokens = [info["token"] for info in self.data_infos]
            if set(index) != set(tokens) or len(tokens) != len(index):
                raise ValueError("CAN features and validation annotations have different tokens.")
            rows = [index[token] for token in tokens]
            timestamps = np.asarray([info["timestamp"] for info in self.data_infos])
            if not np.array_equal(data["timestamps"][rows], timestamps):
                raise ValueError("CAN feature and annotation timestamps differ.")
            motion, controls = data["motion"][rows], data["controls"][rows]
        if motion.shape != (len(tokens), 13) or controls.shape != (len(tokens), 112):
            raise ValueError("Unexpected CAN feature shape.")
        self.frame_context = {
            token: dict(fastworld_scene=info["scene_token"], fastworld_time=float(stamp) / 1e6,
                        fastworld_motion=ego_motion, fastworld_controls=control)
            for token, info, stamp, ego_motion, control in zip(tokens, self.data_infos, timestamps, motion, controls)
        }
        root = Path(kwargs["data_root"])
        for info in self.data_infos:
            for camera in info["cams"].values():
                parts = Path(camera["data_path"]).parts
                relative = Path(*parts[parts.index("samples"):])
                camera["data_path"] = str(root / relative)

    def union2one(self, queue):
        batch = super().union2one(queue)
        for meta in batch["img_metas"].data.values():
            meta.update(self.frame_context[meta["sample_idx"]])
            meta.pop("ego_fut_trajs", None)
            meta.pop("ego_fut_masks", None)
        return batch
