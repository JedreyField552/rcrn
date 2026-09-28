"""Graph-state molecular encoder and route head."""

from __future__ import annotations

import torch

from rcrn.data.molecules import featurise_smiles_list
from rcrn.models.graph_state import MolMambaEncoder, pad_sequences, unpad_sequences
from rcrn.models.route_head import RouteHead, RouteOperator


def _batch(structures: tuple[str, ...] = ("CCO", "c1ccccc1", "CC(=O)O")):
    from rcrn.data.batching import molecule_batch

    identifiers = tuple(f"m{index}" for index in range(len(structures)))
    return molecule_batch(featurise_smiles_list(identifiers, structures))


def test_encoder_emits_one_state_per_molecule() -> None:
    torch.manual_seed(0)
    batch = _batch()
    encoder = MolMambaEncoder(
        atom_feature_dim=batch.atom_features.shape[1],
        bond_feature_dim=batch.bond_features.shape[1],
        hidden_dim=16,
        state_dim=8,
        num_layers=2,
        d_cap=6,
        dropout=0.0,
    )
    embedding = encoder(
        batch.atom_features,
        batch.bond_index,
        batch.bond_features,
        batch.graph_index,
        batch.graph_count,
    )
    assert embedding.graph_state.shape == (3, 6)
    assert embedding.node_states.shape[0] == batch.atom_features.shape[0]
    assert embedding.graph_count == 3


def test_padding_round_trips() -> None:
    values = torch.randn(5, 4)
    graph_index = torch.tensor([0, 0, 1, 2, 2])
    padded, lengths = pad_sequences(values, graph_index, 3)
    restored = unpad_sequences(padded, graph_index, lengths)
    assert torch.allclose(restored, values)
    assert lengths.tolist() == [2, 1, 2]


def test_graphs_do_not_interact() -> None:
    """Encoding one molecule must not depend on the other molecules in the batch."""

    torch.manual_seed(1)
    batch = _batch()
    encoder = MolMambaEncoder(
        atom_feature_dim=batch.atom_features.shape[1],
        bond_feature_dim=batch.bond_features.shape[1],
        hidden_dim=8,
        state_dim=4,
        num_layers=1,
        d_cap=4,
        dropout=0.0,
    ).eval()
    from rcrn.data.batching import subset_batch

    with torch.no_grad():
        together = encoder(
            batch.atom_features,
            batch.bond_index,
            batch.bond_features,
            batch.graph_index,
            batch.graph_count,
        )
        alone_batch = subset_batch(batch, (0,))
        alone = encoder(
            alone_batch.atom_features,
            alone_batch.bond_index,
            alone_batch.bond_features,
            alone_batch.graph_index,
            alone_batch.graph_count,
        )
    assert torch.allclose(together.graph_state[0], alone.graph_state[0], atol=1e-5)


def test_route_operator_rank_cap_changes_the_factorisation() -> None:
    full = RouteOperator(3, 12, 0)
    capped = RouteOperator(3, 12, 1)
    assert full.effective_rank() == 3
    assert capped.effective_rank() == 1
    assert capped.matrix().shape == (3, 12)


def test_route_head_normalises_direction_and_keeps_magnitude() -> None:
    torch.manual_seed(2)
    head = RouteHead(d_cap=8, route_count=3, rank=3, normalise_routes=True)
    state = torch.randn(5, 8)
    fields = head(state)
    norms = torch.linalg.norm(fields.susceptibility, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert fields.magnitude.shape == (5,)
    assert fields.direct_effect.shape == (5,)


def test_route_head_can_skip_normalisation() -> None:
    torch.manual_seed(3)
    head = RouteHead(d_cap=8, route_count=3, rank=3, normalise_routes=False)
    fields = head(torch.randn(4, 8))
    norms = torch.linalg.norm(fields.susceptibility, dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_route_head_rejects_wrong_rank() -> None:
    head = RouteHead(d_cap=4, route_count=2)
    try:
        head(torch.randn(3))
    except ValueError:
        return
    raise AssertionError("a one-dimensional state must be rejected")
