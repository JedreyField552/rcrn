"""Full regime-coupled reaction-norm model.

Ref: Methods, "The regime-route molecular encoder"; "The plaque composition
encoder"; "Reaction-norm coupling and the decline-to-condition gate"; Algorithm 2.

The assembly keeps the two branches separate and makes every removable component a
constructor argument, so a Tier 1 removal or a Tier 3 substitution changes one
argument rather than a code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rcrn.data.batching import MoleculeBatch
from rcrn.models.composition_head import CompositionHead, CompositionOutput
from rcrn.models.coupling import CouplingOutput, ReactionNormCoupling
from rcrn.models.graph_state import MolecularEmbedding, MolMambaEncoder
from rcrn.models.plaque_encoder import PlaqueEmbedding, PlaqueEncoder
from rcrn.models.resistance_head import ResistanceHead, ResistanceOutput
from rcrn.models.route_head import RouteFields, RouteHead
from rcrn.utils.config import Config


@dataclass(slots=True)
class RCRNOutput:
    """Every intermediate the losses, the gate and the certificate consume."""

    molecules: MolecularEmbedding
    plaque: PlaqueEmbedding
    resistance: ResistanceOutput
    composition: CompositionOutput
    route_fields: RouteFields
    coupling: CouplingOutput


class RCRNModel(nn.Module):
    """Two-branch scorer with a reaction-norm coupling and a gated fallback."""

    def __init__(self, config: Config, atom_feature_dim: int, bond_feature_dim: int) -> None:
        super().__init__()
        self.route_count = int(config.require("model.heads.route_count"))
        self.d_cap = int(config.require("model.heads.d_cap"))
        self.coupling_kind = str(config.require("model.coupling.kind"))
        self.composition_kind = str(config.require("model.heads.composition_kind"))
        self.use_molecular = bool(config.get("model.components.molecular_branch", True))
        self.use_plaque = bool(config.get("model.components.plaque_imaging_branch", True))
        self.use_composition = bool(config.get("model.components.plaque_composition_head", True))
        self.normalise_routes = bool(config.get("model.coupling.normalise_routes", True))
        self.interaction_rank = int(config.get("model.coupling.interaction_rank", 3))

        molecular_hidden = int(config.require("model.molecular.hidden_dim"))
        plaque_hidden = int(config.require("model.plaque.widths")[-1])
        dropout = float(config.require("model.molecular.dropout"))

        self.molecular_encoder = MolMambaEncoder(
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            hidden_dim=molecular_hidden,
            state_dim=int(config.require("model.molecular.state_dim")),
            num_layers=int(config.require("model.molecular.num_layers")),
            d_cap=self.d_cap,
            dropout=dropout,
        )
        self.route_head = RouteHead(
            d_cap=self.d_cap,
            route_count=self.route_count,
            rank=self.interaction_rank,
            normalise_routes=self.normalise_routes,
            dropout=dropout,
        )
        self.plaque_encoder = PlaqueEncoder(
            in_channels=int(config.require("model.plaque.in_channels")),
            stem_width=int(config.require("model.plaque.stem_width")),
            depths=tuple(int(v) for v in config.require("model.plaque.depths")),
            widths=tuple(int(v) for v in config.require("model.plaque.widths")),
            patch_size=tuple(int(v) for v in config.require("model.plaque.patch_size")),
            register_tokens=int(config.require("model.plaque.register_tokens")),
            dropout=float(config.require("model.plaque.dropout")),
        )
        self.composition_head = CompositionHead(plaque_hidden, self.composition_kind, dropout)
        self.resistance_head = ResistanceHead(plaque_hidden, self.route_count, dropout)
        self.coupling = ReactionNormCoupling(
            route_count=self.route_count,
            kind=self.coupling_kind,
            dropout=dropout,
        )

    def forward(self, molecules: MoleculeBatch, volumes: Tensor) -> RCRNOutput:
        if self.use_molecular:
            embedding = self.molecular_encoder(
                molecules.atom_features,
                molecules.bond_index,
                molecules.bond_features,
                molecules.graph_index,
                molecules.graph_count,
            )
            route_fields = self.route_head(embedding.graph_state)
        else:
            embedding = self._absent_molecular_embedding(molecules, volumes)
            zeros = torch.zeros(
                molecules.graph_count,
                self.route_count,
                dtype=volumes.dtype,
                device=volumes.device,
            )
            route_fields = RouteFields(
                susceptibility=zeros,
                magnitude=zeros,
                direct_effect=zeros[:, 0],
            )
        plaque = self.plaque_encoder(volumes)
        composition = (
            self.composition_head(plaque.pooled)
            if self.use_composition
            else CompositionOutput(
                index=torch.zeros(volumes.shape[0], dtype=volumes.dtype, device=volumes.device),
                is_probability=False,
            )
        )
        resistance = self._resistance(plaque)
        coupling = self.coupling(
            route_fields,
            resistance.resistance,
            plaque.patch_tokens,
        )
        return RCRNOutput(
            molecules=embedding,
            plaque=plaque,
            resistance=resistance,
            composition=composition,
            route_fields=route_fields,
            coupling=coupling,
        )

    def _resistance(self, plaque: PlaqueEmbedding) -> ResistanceOutput:
        if self.use_plaque:
            return self.resistance_head(plaque.pooled)
        constant = self.resistance_head.route_mean.unsqueeze(0).expand(plaque.pooled.shape[0], -1)
        return ResistanceOutput(resistance=constant, perturbation=torch.zeros_like(constant))

    def _absent_molecular_embedding(
        self, molecules: MoleculeBatch, volumes: Tensor
    ) -> MolecularEmbedding:
        hidden = int(self.molecular_encoder.hidden_dim)
        zeros = torch.zeros(
            molecules.atom_features.shape[0],
            hidden,
            dtype=volumes.dtype,
            device=volumes.device,
        )
        return MolecularEmbedding(
            node_states=zeros,
            graph_state=torch.zeros(
                molecules.graph_count, self.d_cap, dtype=volumes.dtype, device=volumes.device
            ),
            graph_index=molecules.graph_index,
            graph_count=molecules.graph_count,
        )

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def branch_parameters(self) -> dict[str, int]:
        """Parameter count per removable component, for the ablation table."""

        return {
            "molecular_branch": sum(p.numel() for p in self.molecular_encoder.parameters())
            + sum(p.numel() for p in self.route_head.parameters()),
            "plaque_imaging_branch": sum(p.numel() for p in self.plaque_encoder.parameters())
            + sum(p.numel() for p in self.composition_head.parameters())
            + sum(p.numel() for p in self.resistance_head.parameters()),
            "coupling": sum(p.numel() for p in self.coupling.parameters()),
        }

    @torch.no_grad()
    def refresh_route_mean(self, volumes: Tensor, momentum: float = 1.0) -> None:
        """Refresh the route-wise average resistance from a training batch."""

        plaque = self.plaque_encoder(volumes)
        resistance = self.resistance_head(plaque.pooled)
        self.resistance_head.refresh_route_mean(resistance.resistance, momentum)
