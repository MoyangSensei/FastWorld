"""Action-conditioned future observation predictor; no future input at inference."""
import torch
from torch import nn
from torch.nn import functional as F


class CompactObservationEncoder(nn.Module):
    """Fixed full-rank projection prevents learned target collapse.

    Input is frozen RN34/FPN output [cameras, pixels, batch, 256].
    Preserve camera identity and 2x2 spatial layout: 16 tokens x 64 dims.
    """
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(20261101)
        basis, _ = torch.linalg.qr(torch.randn(256, 64, generator=generator))
        self.register_buffer('basis', basis)
        self.spatial_hw = None

    def forward(self, flat, spatial_shapes):
        if self.spatial_hw is None:
            # The eager warmup reads the shape before CUDA Graph capture.
            self.spatial_hw = tuple(spatial_shapes[0].detach().cpu().tolist())
        h, w = self.spatial_hw
        cameras, pixels, batch, channels = flat.shape
        if cameras != 4 or pixels != h*w or channels != 256:
            raise ValueError('Unsupported camera feature shape.')
        grid = flat.permute(2, 0, 3, 1).reshape(batch*4, 256, h, w)
        pooled = F.adaptive_avg_pool2d(grid, (2, 2))
        tokens = pooled.reshape(batch, 4, 256, 4).permute(0, 1, 3, 2).reshape(batch, 16, 256)
        return F.layer_norm(tokens @ self.basis, (64,))


class ActionFutureWorld(nn.Module):
    """Predict z(t+1s) from z(t), ego(t), and the first 1s of an action.

    Only the logged action has an observed target. Alternative actions are
    evaluated by the shared transition and a utility readout; their future
    predictions must not be described as counterfactually verified.
    """
    def __init__(self, bound=0.25):
        super().__init__()
        self.bound = bound
        self.context = nn.Sequential(nn.Linear(16*64+11, 256), nn.LayerNorm(256), nn.GELU())
        self.action = nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, 256))
        self.transition = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 16*64))
        self.utility = nn.Sequential(nn.Linear(16*64, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.utility[-1].weight)
        nn.init.zeros_(self.utility[-1].bias)

    def predict(self, current, ego, actions):
        # 8 poses at 0.5s intervals; observed target is at +1s.
        prefix = actions[..., :2, :]
        a = torch.cat([prefix[..., :2]/10, prefix[..., 2:3].sin(), prefix[..., 2:3].cos()], -1).flatten(-2)
        context = self.context(torch.cat([current.flatten(1), ego], -1))
        hidden = F.gelu(context[:, None] + self.action(a))
        delta = self.transition(hidden).reshape(actions.shape[0], actions.shape[1], 16, 64)
        return current[:, None] + delta

    def forward(self, current, ego, actions, prior):
        future = self.predict(current, ego, actions)
        correction = self.bound*self.utility(future.flatten(-2)).squeeze(-1).tanh()
        return dict(future_world=future, logits=prior+correction, correction=correction)
