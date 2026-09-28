"""Selective state-space kernel.

Ref: Methods, "The regime-route molecular encoder" (a graph and state-space
encoder that runs in linear time with respect to the number of atoms).

The recurrence follows the standard selective parameterisation: a diagonal
state transition, an input-dependent step size, and an input-dependent input and
output map. Real diagonal transition values are initialised on a logarithmic grid,
which keeps the kernel stable for the short sequences a molecule produces.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

NEGATIVE_INFINITY = float("-inf")


class SelectiveStateSpace(nn.Module):
    """One directional selective scan over a token sequence."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.inner_dim = expand * hidden_dim
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(hidden_dim / 16))
        self.in_projection = nn.Linear(hidden_dim, 2 * self.inner_dim, bias=False)
        self.dt_projection = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self.x_projection = nn.Linear(self.inner_dim, self.dt_rank + 2 * state_dim, bias=False)
        self.out_projection = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.skip: Tensor
        self.register_buffer("skip", torch.ones(self.inner_dim))
        self.a_log: Tensor
        self.register_buffer("a_log", self._initial_log_transitions(self.inner_dim, state_dim))
        self.dt_bias: Tensor
        self.register_buffer("dt_bias", self._initial_dt_bias(self.inner_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _initial_log_transitions(inner_dim: int, state_dim: int) -> Tensor:
        grid = torch.arange(1, state_dim + 1, dtype=torch.float32)
        repeated = grid.repeat(inner_dim, 1)
        return torch.log(repeated)

    @staticmethod
    def _initial_dt_bias(inner_dim: int) -> Tensor:
        values = torch.exp(torch.linspace(math.log(1e-3), math.log(1e-1), inner_dim))
        return torch.log(torch.expm1(values))

    def forward(self, values: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Scan a (batch, length, hidden) sequence with an optional ragged mask."""

        if values.dim() != 3:
            raise ValueError("expected a (batch, length, hidden) tensor")
        batch, length, _ = values.shape
        projected = self.in_projection(values)
        gate, stream = projected.chunk(2, dim=-1)
        combined: Tensor = self.x_projection(stream)
        dt_part, input_map, output_map = torch.split(
            combined, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt = torch.nn.functional.softplus(self.dt_projection(dt_part) + self.dt_bias)
        transition = torch.exp(dt.unsqueeze(-1) * -torch.exp(self.a_log))
        state = torch.zeros(
            batch, self.inner_dim, self.state_dim, dtype=values.dtype, device=values.device
        )
        outputs: list[Tensor] = []
        for position in range(length):
            step = stream[:, position]
            decayed = transition[:, position] * state
            injected = (
                dt[:, position].unsqueeze(-1)
                * input_map[:, position].unsqueeze(1)
                * step.unsqueeze(-1)
            )
            state = decayed + injected
            read = torch.einsum("bds,bs->bd", state, output_map[:, position])
            outputs.append(read + self.skip * step)
        stacked = torch.stack(outputs, dim=1)
        if lengths is not None:
            positions = torch.arange(length, device=values.device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)
            stacked = stacked * mask.unsqueeze(-1).to(stacked.dtype)
        gated = stacked * torch.nn.functional.silu(gate)
        projected_out: Tensor = self.out_projection(gated)
        residual: Tensor = self.norm(projected_out + values)
        return residual


class BidirectionalScan(nn.Module):
    """Forward and backward selective scans fused by a learned gate."""

    def __init__(self, hidden_dim: int, state_dim: int, expand: int = 2) -> None:
        super().__init__()
        self.forward_scan = SelectiveStateSpace(hidden_dim, state_dim, expand)
        self.backward_scan = SelectiveStateSpace(hidden_dim, state_dim, expand)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, values: Tensor) -> Tensor:
        forward_states: Tensor = self.forward_scan(values)
        reversed_values = torch.flip(values, dims=[1])
        backward_states: Tensor = torch.flip(self.backward_scan(reversed_values), dims=[1])
        stacked = torch.cat([forward_states, backward_states], dim=-1)
        mixing: Tensor = torch.sigmoid(self.gate(stacked))
        mixed: Tensor = self.fuse(stacked)
        fused: Tensor = mixing * mixed + (1.0 - mixing) * forward_states
        return fused


class StateSpaceStack(nn.Module):
    """A stack of bidirectional scans with pre-norm residuals."""

    def __init__(self, hidden_dim: int, state_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(BidirectionalScan(hidden_dim, state_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: Tensor) -> Tensor:
        hidden = values
        for layer in self.layers:
            hidden = self.dropout(layer(hidden))
        normalized: Tensor = self.norm(hidden)
        return normalized
