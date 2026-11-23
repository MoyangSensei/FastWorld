"""Action-conditioned scoring of the 16 verified trajectories."""

import torch
from torch import nn

from fastworld.models.config_base import PlannerConfig
from fastworld.runtime.release_core_fastworld import build_release_core_fastworld
from fastworld.world_model import ActionFutureWorld, CompactObservationEncoder


class FastWorldPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PlannerConfig()
        self.core = build_release_core_fastworld(self.config)
        self.encoder = CompactObservationEncoder()
        self.world = ActionFutureWorld()
        self.core.body._backbone.register_forward_hook(self._encode_current_state)

    def _encode_current_state(self, module, args, output):
        flat, spatial_shapes, *_ = output
        self.current_state = self.encoder(flat, spatial_shapes)

    def forward(self, features):
        base = self.core(features)
        branches = base["selected_branches"]
        proposals = base["proposals"]
        actions = proposals.gather(1, branches[:, :, None, None].expand(-1, -1, 8, 3))
        prior = base["pred_logit"]
        future = self.world(self.current_state, features["ego_status"][:, -1], actions, prior)

        # Apply the checkpoint's calibrated readout to close decisions.
        leaders = prior.topk(2, dim=1).values
        use_world = (leaders[:, 0] - leaders[:, 1]) <= 0.01
        calibrated = prior - 0.25 * (future["logits"] - prior)
        logits = torch.where(use_world[:, None], calibrated, prior)
        scores = torch.zeros_like(base["pdm_score"]).scatter(1, branches, logits.sigmoid())
        pick = scores.argmax(dim=1)
        trajectory = proposals[torch.arange(len(pick), device=pick.device), pick]
        return {"trajectory": trajectory, "pdm_score": scores}
