"""Segment utilities and graph message passing."""

from __future__ import annotations

import pytest
import torch

from rcrn.models.message_passing import (
    EdgeConditionedConv,
    FragmentAttentionPool,
    GraphAttentionBlock,
    bfs_order,
    degree_of,
    inverse_permutation,
    segment_softmax,
)


def test_segment_softmax_normalises_per_destination() -> None:
    logits = torch.randn(5)
    destination = torch.tensor([0, 0, 1, 2, 2])
    weights = segment_softmax(logits, destination, 3)
    for node in range(3):
        mask = destination == node
        assert float(weights[mask].sum()) == pytest.approx(1.0, abs=1e-6)


def test_degree_counts_both_endpoints() -> None:
    bond_index = torch.tensor([[0, 1], [1, 2]])
    degrees = degree_of(3, bond_index)
    assert degrees.tolist() == [1, 2, 1]


def test_bfs_order_is_a_permutation() -> None:
    bond_index = torch.tensor([[0, 1], [1, 2], [3, 4]])
    batch_index = torch.tensor([0, 0, 0, 1, 1])
    order = bfs_order(5, bond_index, batch_index, 2)
    assert sorted(order.tolist()) == [0, 1, 2, 3, 4]
    inverse = inverse_permutation(order)
    assert inverse[order].tolist() == list(range(5))


def test_edge_conditioned_convolution_changes_node_states() -> None:
    torch.manual_seed(0)
    conv = EdgeConditionedConv(hidden_dim=4, edge_dim=3)
    nodes = torch.randn(4, 4)
    bond_index = torch.tensor([[0, 1], [1, 2], [2, 3]])
    bonds = torch.randn(3, 3)
    output = conv(nodes, bond_index, bonds)
    assert output.shape == nodes.shape
    assert not torch.allclose(output, nodes)


def test_convolution_without_bonds_is_the_self_projection() -> None:
    torch.manual_seed(1)
    conv = EdgeConditionedConv(hidden_dim=4, edge_dim=3)
    nodes = torch.randn(3, 4)
    output = conv(nodes, torch.zeros((0, 2), dtype=torch.long), torch.zeros((0, 3)))
    assert torch.allclose(output, nodes)


def test_attention_weights_do_not_leak_across_destinations() -> None:
    torch.manual_seed(2)
    block = GraphAttentionBlock(hidden_dim=4, heads=2, edge_dim=3).eval()
    nodes = torch.randn(4, 4)
    bond_index = torch.tensor([[0, 1], [2, 3]])
    bonds = torch.randn(2, 3)
    with torch.no_grad():
        first = block(nodes, bond_index, bonds)
        altered = nodes.clone()
        altered[3] += 5.0
        second = block(altered, bond_index, bonds)
    assert torch.allclose(first[:3], second[:3], atol=1e-5)


def test_fragment_attention_pool_returns_one_vector_per_graph() -> None:
    torch.manual_seed(3)
    pool = FragmentAttentionPool(hidden_dim=6, output_dim=2)
    nodes = torch.randn(5, 6)
    graph_index = torch.tensor([0, 0, 1, 1, 1])
    pooled = pool(nodes, graph_index, 2)
    assert pooled.shape == (2, 2)
