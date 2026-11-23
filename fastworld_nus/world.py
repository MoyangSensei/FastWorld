"""Three-step world rollout and utility readout for nuScenes."""

import torch
from torch import nn
from torch.nn import functional as F


class ObservationProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("basis", torch.zeros(256, 64))

    def forward(self, spatial):
        return F.layer_norm(spatial @ self.basis, (64,))


class FutureUtility(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("motion_scale", torch.empty(16))
        self.observed = ObservationProjection()
        self.coarse = nn.Sequential(
            nn.Linear(320, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.attention_query = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 128))
        self.attention_scene = nn.Linear(256, 128)
        self.scene_position = nn.Parameter(torch.empty(36, 128))
        self.cross_attention = nn.MultiheadAttention(128, 4, dropout=0.0)
        self.query_norm = nn.LayerNorm(128)
        self.attention_output = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 1))
        self.temporal_query = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 128))
        self.temporal_tokens = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.LayerNorm(128))
        self.temporal_position = nn.Parameter(torch.empty(36, 128))
        self.temporal_attention = nn.MultiheadAttention(128, 4, dropout=0.0)
        self.temporal_output = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 1))
        self.future_utility = nn.Sequential(nn.Linear(36 * 64 + 64, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 1))
        self.horizon_readout = nn.Sequential(nn.Linear(36 * 64, 96), nn.LayerNorm(96), nn.GELU())
        self.horizon_utility = nn.Sequential(nn.Linear(3 * 96 + 64, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 1))
        self.can_control_prior = nn.Sequential(
            nn.Linear(256 + 48 + 16 + 36, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1),
        )
        self.highrate_control_prior = nn.Sequential(
            nn.Linear(256 + 48 + 16 + 64, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1),
        )
        self.progress_distribution = nn.Sequential(
            nn.Linear(2 * 256 + 16 + 64, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 27),
        )
        self.progress_utility = nn.Sequential(nn.Linear(6, 48), nn.GELU(), nn.Linear(48, 1))
        self.recent_distribution = nn.Sequential(
            nn.Linear(16 + 112, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 27)
        )

    def coarse_scores(self, scene, geometry, motion, past_scene, past_valid):
        pooled = torch.cat((scene.mean(1), motion), -1)[:, None].expand(-1, geometry.shape[1], -1)
        scores = self.coarse(torch.cat((pooled, geometry), -1)).squeeze(-1)
        context = torch.cat((geometry, motion[:, None].expand(-1, geometry.shape[1], -1)), -1)

        query = self.attention_query(context)
        tokens = self.attention_scene(scene) + self.scene_position[None]
        tokens = tokens.transpose(0, 1)
        attended = self.cross_attention(query.transpose(0, 1), tokens, tokens, need_weights=False)[0].transpose(0, 1)
        attention = torch.cat((self.query_norm(query + attended), context), -1)
        scores = scores + self.attention_output(attention).squeeze(-1)

        tokens = self.temporal_tokens(torch.cat((scene, past_scene, scene - past_scene), -1))
        tokens = (tokens + self.temporal_position[None]).transpose(0, 1)
        query = self.temporal_query(context)
        attended = self.temporal_attention(query.transpose(0, 1), tokens, tokens, need_weights=False)[0].transpose(0, 1)
        temporal = self.temporal_output(torch.cat((attended + query, context), -1)).squeeze(-1)
        return scores + temporal * past_valid[:, None]

    def control_prior(self, scene, geometry, motion, control):
        monitor = control[:, :36]
        highrate = control[:, :64]
        pooled = scene.mean(1)
        context = torch.cat((pooled, motion, monitor), -1)
        inputs = torch.cat((geometry, context[:, None].expand(-1, geometry.shape[1], -1)), -1)
        prior = self.can_control_prior(inputs).squeeze(-1) * monitor[:, 8::9].amax(1, keepdim=True)
        context = torch.cat((pooled, motion, highrate), -1)
        inputs = torch.cat((geometry, context[:, None].expand(-1, geometry.shape[1], -1)), -1)
        present = highrate[:, 42::7].amax(1, keepdim=True)
        return prior + self.highrate_control_prior(inputs).squeeze(-1) * present

    def displacement_prior(self, scene, motion, normalized, past_scene, past_valid, control, actions):
        current = scene.mean(1)
        change = (current - past_scene.mean(1)) * past_valid[:, None]
        base = self.progress_distribution(torch.cat((current, change, normalized, control[:, :64]), -1))
        recent = control[:, 64:]
        present = recent[:, [4, 9, 14, 19, 24, 33, 40, 47]].amax(1, keepdim=True)
        residual = self.recent_distribution(torch.cat((normalized, control), -1))
        predicted = (base + present * residual).reshape(-1, 3, 3, 3)
        weight = F.log_softmax(predicted[..., 0], dim=-1).exp()[:, None]
        seconds = torch.arange(1, 4, device=scene.device, dtype=scene.dtype)[None, :, None]
        speed = motion[:, 11].clamp_min(0)[:, None, None]
        acceleration = scene.new_tensor([-.4, 0, .4])[None, None, :]
        mean = (speed * seconds + .5 * acceleration * seconds.square()
                + 4 * seconds * predicted[..., 1].tanh()).clamp_min(0)
        spread = (.3 + seconds * F.softplus(predicted[..., 2].clamp(-5, 5)))[:, None]
        distances = actions.cumsum(2)[:, :, 1::2].norm(dim=-1)
        delta = distances[..., None] - mean[:, None]
        expected = ((delta.abs() + spread * torch.exp(-delta.abs() / spread)) * weight).sum(-1)
        signed = (delta * weight).sum(-1)
        return self.progress_utility(torch.cat((expected, signed), -1) / 12).squeeze(-1)

    def rollout(self, scene, actions, transition):
        batch, count = actions.shape[:2]
        state = scene[:, None].expand(-1, count, -1, -1).reshape(batch * count, 36, 256)
        projected = []
        for second in range(3):
            segment = actions[:, :, second * 2:second * 2 + 2]
            action = torch.cat((segment, torch.zeros_like(actions[:, :, :4])), dim=2).reshape(batch * count, 6, 2)
            state = transition(state, action)
            projected.append(self.observed(state).reshape(batch, count, 36, 64))
        return torch.stack(projected, dim=2)

    def forward(self, scene, candidates, motion, transition, past_scene, past_valid, controls):
        positions = F.pad(candidates.cumsum(-2).flatten(-2) / 32.0, (0, 12))
        actions = F.pad(candidates.flatten(-2) / 8.0, (0, 12))
        geometry = torch.cat((positions, actions), dim=-1)
        normalized = motion / self.motion_scale
        coarse = self.coarse_scores(scene, geometry, normalized, past_scene, past_valid)
        coarse = coarse + self.control_prior(scene, geometry, normalized, controls)
        selected = coarse.topk(16, dim=1).indices
        actions = candidates.gather(1, selected[:, :, None, None].expand(-1, -1, 6, 2))
        states = self.rollout(scene, actions, transition)
        features = torch.cat((geometry.gather(1, selected[:, :, None].expand(-1, -1, 48)),
                              normalized[:, None].expand(-1, 16, -1)), -1)
        prior = self.future_utility(torch.cat((states[:, :, -1].flatten(-2), features), -1)).squeeze(-1)
        horizon = self.horizon_readout(states.flatten(-2)).flatten(-2)
        utility = prior + self.horizon_utility(torch.cat((horizon, features), -1)).squeeze(-1)
        utility = utility + self.displacement_prior(scene, motion, normalized, past_scene, past_valid, controls, actions)
        refined = coarse.gather(1, selected) + 12.0 * (utility * (2.0 / 12.0)).tanh()
        winner = selected.gather(1, refined.argmax(1, keepdim=True)).squeeze(1)
        return candidates[torch.arange(len(winner), device=winner.device), winner]
