"""Model configuration for the NAVSIM v1 NAVTEST checkpoint."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PlannerConfig:
    image_architecture: str = "resnet34"
    num_cams: int = 4
    ref_num: int = 4
    proposal_num: int = 64
    num_poses: int = 8
    traj_proposal_query: bool = True
    num_points_in_pillar: int = 4
    point_cloud_range: ClassVar = (-32, -32, -2.0, 32, 32, 6.0)
    half_length: float = 2.838
    half_width: float = 1.1485 + 0.1
    rear_axle_to_center: float = 1.461
    lidar_height: float = 0
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_dropout: float = 0
    num_bev_layers: int = 1
