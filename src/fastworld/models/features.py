"""Current four-camera CPU feature builder, matching the locked release."""
import torch
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
from .bevformer.bev_feature_build_fastworld import _get_bev_feature_fastworld
class FastWorldFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,config):self._config=config
    def get_unique_name(self):return 'fastworld_resnet_feature'
    def compute_features(self,agent_input):
        features=_get_bev_feature_fastworld(agent_input,return_images=True)
        features['ego_status']=torch.stack([torch.cat([torch.tensor(s.ego_pose),torch.tensor(s.ego_velocity),torch.tensor(s.ego_acceleration),torch.tensor(s.driving_command)]) for s in agent_input.ego_statuses]).float()
        return features
