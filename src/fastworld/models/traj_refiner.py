import torch.nn as nn

from .bevformer.bev_refiner import Bev_refiner
from .bevformer.transformer_decoder import MLP
from .config_base import PlannerConfig


class Traj_refiner(nn.Module):
    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.poses_num = config.num_poses
        self.state_size = 3
        self.Bev_refiner = Bev_refiner(config, config.proposal_num, self.poses_num, config.traj_proposal_query)
        self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn, self.state_size)

    def forward(self, bev_feature, proposal_list, image_feature):
        proposals = self.traj_decoder(bev_feature).reshape(
            bev_feature.shape[0], -1, self.poses_num, self.state_size
        )
        proposal_list.append(proposals)
        bev_feature = self.Bev_refiner(proposals, bev_feature, image_feature)
        return bev_feature, proposal_list
