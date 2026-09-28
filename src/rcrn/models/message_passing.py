"""Graph message passing and segment utilities for the molecular branch.

Ref: Methods, "The regime-route molecular encoder".

Attention and convolution are implemented over the flat edge list with segment
softmax, so a batch of molecules of different sizes is one pair of index
operations rather than a dense adjacency per molecule.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn


def segment_softmax(logits: Tensor, destination: Tensor, node_count: int) -> Tensor:
    """Softmax over the incoming edges of each destination node."""

    maxima = torch.full((node_count,), float("-inf"), dtype=logits.dtype, device=logits.device)
    maxima = maxima.scatter_reduce(0, destination, logits, reduce="amax", include_self=True)
    shifted = logits - maxima[destination]
    exponentials = torch.exp(shifted)
    totals = torch.zeros(node_count, dtype=logits.dtype, device=logits.device)
    totals = totals.index_add(0, destination, exponentials)
    return exponentials / (totals[destination] + 1e-9)


def degree_of(node_count: int, bond_index: Tensor) -> Tensor:
    """Undirected degree of every node in a flat batch."""

    if bond_index.numel() == 0:
        return torch.zeros(node_count, dtype=torch.long, device=bond_index.device)
    sources = bond_index[:, 0]
    destinations = bond_index[:, 1]
    degrees = torch.zeros(node_count, dtype=torch.long, device=bond_index.device)
    degrees = degrees.index_add(0, sources, torch.ones_like(sources))
    return degrees.index_add(0, destinations, torch.ones_like(destinations))


def bfs_order(node_count: int, bond_index: Tensor, batch_index: Tensor, graph_count: int) -> Tensor:
    """Canonical breadth-first node order per graph, so a scan follows bond paths."""

    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    if bond_index.numel() > 0:
        for source, destination in bond_index.tolist():
            adjacency[source].append(destination)
            adjacency[destination].append(source)
    members: dict[int, list[int]] = {graph: [] for graph in range(graph_count)}
    for node, graph in enumerate(batch_index.tolist()):
        members[graph].append(node)
    visited: set[int] = set()
    sequence: list[int] = []
    for graph in range(graph_count):
        nodes = sorted(members[graph])
        if not nodes:
            continue
        root = nodes[0]
        frontier = deque([root])
        visited.add(root)
        while frontier:
            node = frontier.popleft()
            sequence.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    frontier.append(neighbour)
        for node in nodes:
            if node not in visited:
                visited.add(node)
                sequence.append(node)
    return torch.tensor(sequence, dtype=torch.long, device=bond_index.device)


def inverse_permutation(permutation: Tensor) -> Tensor:
    """Inverse of a permutation vector."""

    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


class EdgeConditionedConv(nn.Module):
    """Message passing with an edge-conditioned weight and a degree-normalised sum."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.source_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, bond_index: Tensor, bond_features: Tensor) -> Tensor:
        node_count = nodes.shape[0]
        if bond_index.numel() == 0:
            return nodes
        sources = bond_index[:, 0]
        destinations = bond_index[:, 1]
        edge_weights: Tensor = self.edge_projection(bond_features).reshape(
            bond_index.shape[0], nodes.shape[1], nodes.shape[1]
        )
        messages = torch.einsum("ed,edf->ef", self.source_projection(nodes)[sources], edge_weights)
        aggregated = torch.zeros_like(nodes)
        aggregated = aggregated.index_add(0, destinations, messages)
        degrees = degree_of(node_count, bond_index).to(nodes.dtype).unsqueeze(-1) + 1.0
        updated: Tensor = self.self_projection(nodes) + aggregated / degrees
        normalized: Tensor = self.norm(updated)
        return self.dropout(normalized)


class GraphAttentionBlock(nn.Module):
    """Multi-head attention over the bond list with an additive edge bias."""

    def __init__(self, hidden_dim: int, heads: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden width must divide by the head count")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, bond_index: Tensor, bond_features: Tensor) -> Tensor:
        if bond_index.numel() == 0:
            return nodes
        node_count = nodes.shape[0]
        sources = bond_index[:, 0]
        destinations = bond_index[:, 1]
        queries: Tensor = self.query(nodes).reshape(node_count, self.heads, self.head_dim)
        keys: Tensor = self.key(nodes).reshape(node_count, self.heads, self.head_dim)
        values: Tensor = self.value(nodes).reshape(node_count, self.heads, self.head_dim)
        bias: Tensor = self.edge_bias(bond_features)
        logits = (queries[sources] * keys[destinations]).sum(-1) / (self.head_dim**0.5)
        logits = logits + bias
        attention = torch.empty_like(logits)
        for head in range(self.heads):
            attention[:, head] = segment_softmax(logits[:, head], destinations, node_count)
        attention = self.dropout(attention)
        messages = values[sources] * attention.unsqueeze(-1)
        aggregated = torch.zeros(node_count, self.heads, self.head_dim, device=nodes.device)
        aggregated = aggregated.index_add(0, destinations, messages)
        merged: Tensor = self.output(aggregated.reshape(node_count, self.hidden_dim))
        residual: Tensor = self.norm(merged + nodes)
        return residual


class FragmentAttentionPool(nn.Module):
    """Attention pooling of node states within one molecule."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, nodes: Tensor, graph_index: Tensor, graph_count: int) -> Tensor:
        logits: Tensor = self.score(nodes).squeeze(-1)
        maxima = torch.full((graph_count,), float("-inf"), dtype=nodes.dtype, device=nodes.device)
        maxima = maxima.scatter_reduce(0, graph_index, logits, reduce="amax", include_self=True)
        exponentials = torch.exp(logits - maxima[graph_index])
        totals = torch.zeros(graph_count, dtype=nodes.dtype, device=nodes.device)
        totals = totals.index_add(0, graph_index, exponentials)
        weights = exponentials / (totals[graph_index] + 1e-9)
        pooled = torch.zeros(graph_count, nodes.shape[1], dtype=nodes.dtype, device=nodes.device)
        pooled = pooled.index_add(0, graph_index, nodes * weights.unsqueeze(-1))
        projected: Tensor = self.projection(pooled)
        return projected
