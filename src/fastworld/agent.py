"""FastWorld inference agent for NAVSIM v1."""

import sys
from pathlib import Path

import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory

from fastworld.planner import FastWorldPlanner
from fastworld.runtime.fixed_query_cross_backend_fastworld import use_fixed_query_cross_backend_fastworld
from fastworld.runtime.release_frontend_fastworld import GPUResNetFeatureBuilderFastWorld, current_sensors_fastworld
from fastworld.runtime.release_graph_constants_fastworld import use_graph_safe_constants_fastworld
from fastworld.runtime.score_graph_backend_fastworld import FeatureGraphPlanner


class FastWorldAgent(AbstractAgent):
    def __init__(self, checkpoint_path):
        super().__init__()
        self._checkpoint_path = Path(checkpoint_path)

    def name(self):
        return "fastworld_navsim_v1"

    def get_sensor_config(self):
        return current_sensors_fastworld()

    def get_feature_builders(self):
        return [GPUResNetFeatureBuilderFastWorld(self._config)]

    def initialize(self):
        if not sys.flags.optimize:
            raise RuntimeError("Launch NAVSIM with python -O for CUDA Graph inference.")
        if not torch.cuda.is_available():
            raise RuntimeError("FastWorld requires a CUDA-enabled GPU.")

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        saved = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("format") != "fastworld_navsim_v1_inference":
            raise ValueError("Expected the FastWorld NAVSIM v1 inference checkpoint.")
        planner = FastWorldPlanner()
        planner.load_state_dict(saved["state_dict"], strict=True)
        planner = planner.cuda().eval().requires_grad_(False)

        # The existing graph preparation functions operate on a carrier for the
        # selective trajectory core. The future-state model stays outside that core.
        self._pad_model = planner.core
        if use_fixed_query_cross_backend_fastworld(self) != 1:
            raise RuntimeError("Unexpected number of cross-attention modules.")
        use_graph_safe_constants_fastworld(self)
        self._config = planner.config
        self._pad_model = FeatureGraphPlanner(planner)
        self.eval()

    def forward(self, features):
        return self._pad_model(features)

    @torch.inference_mode()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        features = self.get_feature_builders()[0].compute_features(agent_input)
        batch = {key: value.unsqueeze(0).cuda() for key, value in features.items()}
        poses = self(batch)["trajectory"].squeeze(0).cpu().numpy()
        return Trajectory(poses)
