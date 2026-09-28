"""Hybrid graph and state-space molecular encoder.

Ref: Methods, "The regime-route molecular encoder"; Algorithm 2 step 1.

Each block alternates an edge-conditioned convolution, an attention block and a
state-space scan taken in breadth-first atom order. The scan is the only operation
whose cost grows with the candidate list, and it is linear in the atom count, which
is what the study's added-inference-cost constraint requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rcrn.models.message_passing import (
    EdgeConditionedConv,
    FragmentAttentionPool,
    GraphAttentionBlock,
    bfs_order,
    inverse_permutation,
)
from rcrn.models.state_space import StateSpaceStack


@dataclass(slots=True)
class MolecularEmbedding:
    """Node states, the per-molecule coupling state and the graph index."""

    node_states: Tensor
    graph_state: Tensor
    graph_index: Tensor
    graph_count: int


def pad_sequences(values: Tensor, graph_index: Tensor, graph_count: int) -> tuple[Tensor, Tensor]:
    """Pack a flat node axis into a padded (graph, length, hidden) block."""

    hidden = values.shape[1]
    lengths = torch.zeros(graph_count, dtype=torch.long, device=values.device)
    lengths = lengths.index_add(0, graph_index, torch.ones_like(graph_index))
    longest = int(lengths.max().item()) if graph_count else 0
    padded = torch.zeros(
        graph_count, max(longest, 1), hidden, dtype=values.dtype, device=values.device
    )
    offsets = torch.cumsum(lengths, dim=0) - lengths
    positions = torch.arange(values.shape[0], device=values.device) - offsets[graph_index]
    padded[graph_index, positions] = values
    return padded, lengths


def unpad_sequences(padded: Tensor, graph_index: Tensor, lengths: Tensor) -> Tensor:
    """Inverse of :func:`pad_sequences`."""

    offsets = torch.cumsum(lengths, dim=0) - lengths
    positions = torch.arange(graph_index.numel(), device=padded.device) - offsets[graph_index]
    return padded[graph_index, positions]


class MolMambaBlock(nn.Module):
    """One graph-state block: convolution, attention, then a breadth-first scan."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        edge_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = EdgeConditionedConv(hidden_dim, edge_dim, dropout)
        self.attention = GraphAttentionBlock(hidden_dim, heads, edge_dim, dropout)
        self.scan = StateSpaceStack(hidden_dim, state_dim, layers=1, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: Tensor,
        bond_index: Tensor,
        bond_features: Tensor,
        graph_index: Tensor,
        graph_count: int,
    ) -> Tensor:
        convolved: Tensor = self.conv(nodes, bond_index, bond_features)
        attended: Tensor = self.attention(convolved, bond_index, bond_features)
        permutation = bfs_order(nodes.shape[0], bond_index, graph_index, graph_count)
        inverse = inverse_permutation(permutation)
        reordered = attended[permutation]
        reordered_index = graph_index[permutation]
        padded, lengths = pad_sequences(reordered, reordered_index, graph_count)
        scanned: Tensor = self.scan(padded)
        restored = unpad_sequences(scanned, reordered_index, lengths)
        original_order = restored[inverse]
        residual: Tensor = self.norm(attended + original_order)
        return residual


class MolMambaEncoder(nn.Module):
    """Atom and fragment encoder that emits the coupling state of every molecule."""

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int,
        state_dim: int,
        num_layers: int,
        d_cap: int,
        dropout: float,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        self.bond_embedding = nn.Linear(bond_feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            MolMambaBlock(hidden_dim, state_dim, hidden_dim, heads, dropout)
            for _ in range(num_layers)
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.pool = FragmentAttentionPool(hidden_dim, d_cap)
        self.fragment_mix = nn.Linear(d_cap, d_cap)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_features: Tensor,
        bond_index: Tensor,
        bond_features: Tensor,
        graph_index: Tensor,
        graph_count: int,
    ) -> MolecularEmbedding:
        nodes: Tensor = self.atom_embedding(atom_features)
        bonds: Tensor = self.bond_embedding(bond_features)
        for block in self.blocks:
            nodes = block(nodes, bond_index, bonds, graph_index, graph_count)
        normalized: Tensor = self.node_norm(nodes)
        pooled: Tensor = self.pool(normalized, graph_index, graph_count)
        mixed: Tensor = self.fragment_mix(torch.tanh(pooled))
        graph_state: Tensor = self.dropout(mixed)
        return MolecularEmbedding(
            node_states=normalized,
            graph_state=graph_state,
            graph_index=graph_index,
            graph_count=graph_count,
        )
