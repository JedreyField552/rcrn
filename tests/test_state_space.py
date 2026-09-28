"""Selective state-space kernel."""

from __future__ import annotations

import torch
from torch import nn

from rcrn.models.state_space import BidirectionalScan, SelectiveStateSpace, StateSpaceStack


def test_scan_preserves_shape() -> None:
    block = SelectiveStateSpace(hidden_dim=8, state_dim=4)
    values = torch.randn(2, 5, 8)
    assert block(values).shape == values.shape


def test_scan_is_causal_in_the_forward_direction() -> None:
    """Changing a later step must not change an earlier output."""

    torch.manual_seed(0)
    block = SelectiveStateSpace(hidden_dim=6, state_dim=3).eval()
    values = torch.randn(1, 6, 6)
    altered = values.clone()
    altered[:, 4:] += 3.0
    with torch.no_grad():
        first = block(values)
        second = block(altered)
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6)
    assert not torch.allclose(first[:, 4:], second[:, 4:])


def test_length_mask_suppresses_the_scan_contribution() -> None:
    """A zero-padded position stays zero: the mask removes the scan's contribution there."""

    torch.manual_seed(1)
    block = SelectiveStateSpace(hidden_dim=4, state_dim=2).eval()
    values = torch.randn(1, 5, 4)
    values[:, 3:] = 0.0
    lengths = torch.tensor([3])
    with torch.no_grad():
        output = block(values, lengths)
    assert torch.allclose(output[0, 3:], torch.zeros_like(output[0, 3:]), atol=1e-6)
    assert not torch.allclose(output[0, :3], torch.zeros_like(output[0, :3]), atol=1e-6)


def test_sequence_is_position_sensitive_before_gating() -> None:
    """The convolution-free scan itself must not be permutation invariant."""

    torch.manual_seed(2)
    scan = BidirectionalScan(hidden_dim=6, state_dim=3).eval()
    values = torch.randn(1, 5, 6)
    with torch.no_grad():
        straight = scan(values)
        shuffled = scan(values.flip(1)).flip(1)
    assert not torch.allclose(straight, shuffled, atol=1e-4)


def test_stack_keeps_the_hidden_width_and_carries_gradients() -> None:
    torch.manual_seed(3)
    stack = StateSpaceStack(hidden_dim=6, state_dim=3, layers=2, dropout=0.0)
    values = torch.randn(2, 4, 6, requires_grad=True)
    output = stack(values)
    assert output.shape == values.shape
    output.sum().backward()
    assert values.grad is not None and float(values.grad.abs().sum()) > 0.0


def test_rejects_wrong_rank() -> None:
    block = SelectiveStateSpace(hidden_dim=4, state_dim=2)
    try:
        block(torch.randn(4, 4))
    except ValueError:
        return
    raise AssertionError("a two-dimensional input must be rejected")


def test_inner_width_follows_the_expansion_factor() -> None:
    block = SelectiveStateSpace(hidden_dim=5, state_dim=3, expand=3)
    assert isinstance(block.in_projection, nn.Linear)
    assert block.inner_dim == 15
    assert block.a_log.shape == (15, 3)
