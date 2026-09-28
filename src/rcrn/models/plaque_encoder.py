"""Multi-contrast plaque encoder.

Ref: Methods, "The plaque composition encoder" (a three-dimensional multi-contrast
encoder with multi-scale stages and register tokens).

The stem patchifies the volume, four stages mix spatial context with the
convolutional block, and the resulting tokens pass through a transformer stage
with register tokens prepended. Register tokens give the composition and
resistance heads a slot that is not tied to any one plaque voxel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class PlaqueEmbedding:
    """Patch tokens, register tokens and the pooled descriptor of one plaque."""

    patch_tokens: Tensor
    register_tokens: Tensor
    pooled: Tensor


class GlobalResponseNorm(nn.Module):
    """Channel-wise response normalisation across the spatial axes."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma: Tensor
        self.register_buffer("gamma", torch.zeros(1, channels, 1, 1, 1))
        self.beta: Tensor
        self.register_buffer("beta", torch.zeros(1, channels, 1, 1, 1))

    def forward(self, values: Tensor) -> Tensor:
        energy = torch.sqrt(torch.sum(values * values, dim=(2, 3, 4), keepdim=True) + 1e-6)
        normalised = energy / (energy.mean(dim=1, keepdim=True) + 1e-6)
        return values + self.gamma * (values * normalised) + self.beta


class ConvNeXtBlock(nn.Module):
    """Depthwise spatial mixing followed by an inverted bottleneck."""

    def __init__(self, channels: int, expansion: int = 4, kernel: int = 5) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size=kernel, padding=kernel // 2, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Linear(channels, expansion * channels)
        self.contract = nn.Linear(expansion * channels, channels)
        self.response = GlobalResponseNorm(channels)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        mixed: Tensor = self.depthwise(values)
        normalized: Tensor = self.norm(mixed)
        channels_last = normalized.permute(0, 2, 3, 4, 1)
        expanded: Tensor = self.activation(self.expand(channels_last))
        contracted: Tensor = self.contract(expanded)
        restored = contracted.permute(0, 4, 1, 2, 3)
        gated: Tensor = self.response(restored)
        residual: Tensor = values + gated
        return residual


class TransformerBlock(nn.Module):
    """Pre-norm transformer block over patch and register tokens."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized: Tensor = self.norm1(tokens)
        attended: Tensor = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        after_attention: Tensor = tokens + self.dropout(attended)
        normalized_2: Tensor = self.norm2(after_attention)
        after_mlp: Tensor = after_attention + self.dropout(self.mlp(normalized_2))
        return after_mlp


class PlaqueEncoder(nn.Module):
    """Encode a multi-contrast plaque volume into patch and register states."""

    def __init__(
        self,
        in_channels: int,
        stem_width: int,
        depths: tuple[int, ...],
        widths: tuple[int, ...],
        patch_size: tuple[int, int, int],
        register_tokens: int,
        dropout: float,
        transformer_layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if len(depths) != len(widths):
            raise ValueError("depth and width schedules must agree")
        self.patch_size = patch_size
        self.register_count = register_tokens
        self.stem = nn.Conv3d(in_channels, stem_width, kernel_size=patch_size, stride=patch_size)
        self.stem_norm = nn.GroupNorm(1, stem_width)
        stages: list[nn.Module] = []
        downsamples: list[nn.Module] = []
        current = stem_width
        for depth, width in zip(depths, widths, strict=True):
            blocks = nn.Sequential(*[ConvNeXtBlock(width) for _ in range(depth)])
            if current != width:
                downsamples.append(
                    nn.Sequential(
                        nn.GroupNorm(1, current), nn.Conv3d(current, width, kernel_size=1)
                    )
                )
            else:
                downsamples.append(nn.Identity())
            stages.append(blocks)
            current = width
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.project = nn.Linear(current, widths[-1])
        self.register: Tensor
        self.register_buffer("register", torch.zeros(1, register_tokens, widths[-1]))
        self.blocks = nn.ModuleList(
            TransformerBlock(widths[-1], heads, dropout) for _ in range(transformer_layers)
        )
        self.norm = nn.LayerNorm(widths[-1])
        self.pool_score = nn.Linear(widths[-1], 1)

    def forward(self, volumes: Tensor) -> PlaqueEmbedding:
        if volumes.dim() != 5:
            raise ValueError("expected a (batch, channel, depth, height, width) volume")
        features: Tensor = self.stem_norm(self.stem(volumes))
        for blocks, downsample in zip(self.stages, self.downsamples, strict=True):
            features = downsample(features)
            features = blocks(features)
        batch = features.shape[0]
        channels_last = features.permute(0, 2, 3, 4, 1).reshape(batch, -1, features.shape[1])
        tokens: Tensor = self.project(channels_last)
        registers = self.register.expand(batch, -1, -1)
        combined = torch.cat([registers, tokens], dim=1)
        hidden = combined
        for block in self.blocks:
            hidden = block(hidden)
        normalized: Tensor = self.norm(hidden)
        register_states = normalized[:, : self.register_count]
        patch_states = normalized[:, self.register_count :]
        pooled = self._attention_pool(patch_states)
        return PlaqueEmbedding(
            patch_tokens=patch_states,
            register_tokens=register_states,
            pooled=pooled,
        )

    def _attention_pool(self, patch_tokens: Tensor) -> Tensor:
        logits: Tensor = self.pool_score(patch_tokens).squeeze(-1)
        weights: Tensor = torch.softmax(logits, dim=-1)
        pooled: Tensor = torch.sum(patch_tokens * weights.unsqueeze(-1), dim=1)
        return pooled
