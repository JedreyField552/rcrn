"""Plaque encoder, composition head and resistance head."""

from __future__ import annotations

import pytest
import torch

from rcrn.models.composition_head import CompositionHead
from rcrn.models.plaque_encoder import ConvNeXtBlock, GlobalResponseNorm, PlaqueEncoder
from rcrn.models.resistance_head import ResistanceHead


def _encoder() -> PlaqueEncoder:
    return PlaqueEncoder(
        in_channels=4,
        stem_width=8,
        depths=(1, 1, 1, 1),
        widths=(8, 8, 16, 16),
        patch_size=(1, 2, 2),
        register_tokens=2,
        dropout=0.0,
        transformer_layers=1,
        heads=2,
    )


def test_encoder_emits_patch_and_register_tokens() -> None:
    torch.manual_seed(0)
    encoder = _encoder()
    volume = torch.randn(2, 4, 4, 16, 16)
    embedding = encoder(volume)
    assert embedding.register_tokens.shape == (2, 2, 16)
    assert embedding.patch_tokens.shape[0] == 2
    assert embedding.pooled.shape == (2, 16)


def test_encoder_pooled_state_has_gradients() -> None:
    torch.manual_seed(1)
    encoder = _encoder()
    volume = torch.randn(1, 4, 4, 16, 16, requires_grad=True)
    embedding = encoder(volume)
    embedding.pooled.sum().backward()
    assert volume.grad is not None and float(volume.grad.abs().sum()) > 0.0


def test_register_tokens_mean_pool_is_not_a_single_patch() -> None:
    torch.manual_seed(2)
    encoder = _encoder().eval()
    volume = torch.randn(1, 4, 4, 16, 16)
    with torch.no_grad():
        embedding = encoder(volume)
    distances = torch.linalg.norm(embedding.pooled[:, None, :] - embedding.patch_tokens, dim=-1)
    assert float(distances.min()) > 1e-4


def test_encoder_rejects_a_four_dimensional_input() -> None:
    with pytest.raises(ValueError):
        _encoder()(torch.randn(4, 8, 8))


def test_response_norm_preserves_shape() -> None:
    norm = GlobalResponseNorm(6)
    values = torch.randn(2, 6, 3, 3, 3)
    assert norm(values).shape == values.shape


def test_convnext_block_transforms_and_carries_gradients() -> None:
    torch.manual_seed(3)
    block = ConvNeXtBlock(6)
    values = torch.randn(1, 6, 3, 3, 3, requires_grad=True)
    output = block(values)
    assert output.shape == values.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert values.grad is not None and float(values.grad.abs().sum()) > 0.0


def test_composition_head_is_bounded() -> None:
    head = CompositionHead(hidden_dim=8, kind="composition_index")
    index = head(torch.randn(4, 8)).index
    assert index.shape == (4,)
    assert bool(((index > 0.0) & (index < 1.0)).all())


def test_vulnerability_head_reports_a_probability() -> None:
    head = CompositionHead(hidden_dim=8, kind="vulnerability_classifier")
    output = head(torch.randn(3, 8))
    assert output.is_probability is True


def test_resistance_head_is_positive_and_centred() -> None:
    torch.manual_seed(4)
    head = ResistanceHead(hidden_dim=8, route_count=3)
    output = head(torch.randn(5, 8))
    assert bool((output.resistance > 0.0).all())
    assert output.perturbation.shape == (5, 3)
    assert torch.allclose(output.resistance - head.route_mean, output.perturbation)


def test_route_mean_can_be_refreshed() -> None:
    head = ResistanceHead(hidden_dim=4, route_count=2)
    before = head.route_mean.clone()
    table = ResistanceHead(hidden_dim=4, route_count=2)
    with torch.no_grad():
        table.head.weight.copy_(head.head.weight)
        table.head.bias.copy_(head.head.bias)
    head.refresh_route_mean(torch.tensor([[2.0, 4.0], [4.0, 8.0]]), momentum=1.0)
    assert not torch.allclose(before, head.route_mean)
    assert torch.allclose(head.route_mean, torch.tensor([3.0, 6.0]), atol=1e-5)
