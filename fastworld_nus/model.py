"""The fixed nuScenes FastWorld planner and its two-frame LAW interface."""

import numpy as np
import torch
from mmdet.models import DETECTORS, HEADS
from projects.mmdet3d_plugin.LAW.LAW import LAW
from projects.mmdet3d_plugin.LAW.dense_heads.waypoint_query_decoder import WaypointHead, inverse_sigmoid
from torch import nn

from .world import FutureUtility


class ProposalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Embedding(64, 256)
        self.context = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256))
        self.offset = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 12))

    def forward(self, waypoints, spatial):
        query = self.query.weight[None].expand(len(spatial), -1, -1)
        features = self.context(torch.cat((query, spatial.mean(1)[:, None].expand_as(query)), dim=-1))
        offsets = self.offset(features).reshape(len(waypoints), 64, 6, 2)
        return waypoints[:, None] + offsets


@HEADS.register_module()
class FastWorldNuScenesHead(WaypointHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fastworld = ProposalHead()
        self.motion_ranker = FutureUtility()
        self.previous = None
        self.current_tokens = None

    def img_position_embeding(self, img_feats, img_metas):
        """Use the CPU calibration inverse from the evaluated LAW implementation."""
        batch, views, _, height, width = img_feats.shape
        length = views * height * width
        locations = self.prepare_location(img_metas, img_feats)
        pad_height, pad_width, _ = img_metas[0]["pad_shape"][0]
        locations[..., 0] *= pad_width
        locations[..., 1] *= pad_height

        depth = self.coords_d.shape[0]
        pixels = locations.detach().view(batch, length, 1, 2).repeat(1, 1, depth, 1)
        depth_coords = self.coords_d.view(1, 1, depth, 1).repeat(batch, length, 1, 1)
        coordinates = torch.cat((pixels, depth_coords), dim=-1)
        coordinates = torch.cat((coordinates, torch.ones_like(coordinates[..., :1])), -1)
        coordinates[..., :2] *= torch.maximum(coordinates[..., 2:3], torch.ones_like(coordinates[..., 2:3]) * 1e-5)

        lidar_to_image = torch.from_numpy(np.stack(img_metas[0]["lidar2img"])).float()[:views]
        image_to_lidar = torch.linalg.inv(lidar_to_image).to(img_feats.device)
        matrices = image_to_lidar.view(views, 1, 1, 4, 4).repeat(batch, height * width, depth, 1, 1)
        points = torch.matmul(matrices.view(batch, length, depth, 4, 4), coordinates.unsqueeze(-1)).squeeze(-1)[..., :3]
        low, high = self.position_range[:3], self.position_range[3:6]
        points[..., :3] = (points[..., :3] - low) / (high - low)
        return self.position_encoder(inverse_sigmoid(points.reshape(batch, -1, depth * 3)))

    def forward(self, img_feat, img_metas, ego_info=None, is_test=False):
        waypoints, spatial, future = super().forward(img_feat, img_metas, ego_info, is_test)
        tokens = [meta["sample_idx"] for meta in img_metas]
        if tokens != self.current_tokens:
            self.previous = (spatial, img_metas)
            return waypoints, spatial, future

        past = spatial
        valid = spatial.new_zeros(len(spatial))
        if self.previous is not None:
            past, metadata = self.previous
            valid = spatial.new_tensor([
                old["fastworld_scene"] == new["fastworld_scene"]
                and .85 <= new["fastworld_time"] - old["fastworld_time"] <= 1.15
                for old, new in zip(metadata, img_metas)
            ])
        candidates = self.fastworld(waypoints, spatial)
        causal = spatial.new_tensor(np.stack([meta["fastworld_motion"] for meta in img_metas]))
        command = torch.stack([meta["ego_fut_cmd"].reshape(-1) for meta in img_metas]).to(spatial.device)
        motion = torch.cat((causal, command), dim=1)
        controls = spatial.new_tensor(np.stack([meta["fastworld_controls"] for meta in img_metas]))
        trajectory = self.motion_ranker(spatial, candidates, motion, self.wm_prediction, past, valid, controls)
        return trajectory, spatial, future


@DETECTORS.register_module()
class FastWorldNuScenes(LAW):
    def predict_trajectory(self, images, metadata):
        """Return six local XY increments at 0.5-second intervals."""
        head = self.pts_bbox_head
        last = images.size(1) - 1
        current = [sample[last] for sample in metadata]
        head.prev_view_feat = None
        head.previous = None
        # Encode the history first. The upstream queue can repeat the current
        # frame at a scene boundary, where it contributes no temporal correction.
        head.current_tokens = None
        try:
            self.obtain_history_feat(images[:, :-1], metadata, is_test=True)
            head.current_tokens = [sample["sample_idx"] for sample in current]
            features = self.extract_feat(img=images[:, -1], img_metas=current)[0]
            return head(features, current)[0]
        finally:
            head.current_tokens = None
            head.previous = None
