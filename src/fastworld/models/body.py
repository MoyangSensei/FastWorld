"""Only the RN34/FPN and shared trajectory decoder used by the released planner."""
import torch
from torch import nn
from .bevformer.resnet_image_encoder_fastworld import ResNetImgEncoderFastWorld
from .traj_refiner import Traj_refiner
from .bevformer.transformer_decoder import MLP
class NativeScoreContainerFastWorld(nn.Module):
    def __init__(self,config):
        super().__init__();self.double_score=False;self.pred_score=MLP(config.tf_d_model,config.tf_d_ffn,6)
class PlannerBodyFastWorld(nn.Module):
    def __init__(self,config):
        super().__init__();self._config=config
        self._backbone=ResNetImgEncoderFastWorld(config)
        self.hist_encoding=nn.Linear(11,config.tf_d_model)
        self.init_feature=nn.Embedding(config.num_poses*config.proposal_num,config.tf_d_model)
        shared=Traj_refiner(config);self._trajectory_head=nn.ModuleList([shared for _ in range(config.ref_num)])
        self.scorer=NativeScoreContainerFastWorld(config)
