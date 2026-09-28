"""Model components: molecular graph-state encoder, plaque encoder, coupling and gate."""

from __future__ import annotations

from rcrn.models.composition_head import CompositionHead, CompositionOutput
from rcrn.models.coupling import (
    COUPLING_KINDS,
    CouplingOutput,
    ReactionNormCoupling,
    route_share,
)
from rcrn.models.gate import (
    DeclineToConditionGate,
    GateCalibration,
    GateDecision,
    calibrate_threshold,
    false_conditioning_rate,
)
from rcrn.models.graph_state import MolecularEmbedding, MolMambaBlock, MolMambaEncoder
from rcrn.models.interaction import (
    InteractionEstimator,
    InteractionSurfaces,
    interaction_statistic,
    perturbation_noise,
    shrinkage_covariance,
)
from rcrn.models.message_passing import (
    EdgeConditionedConv,
    FragmentAttentionPool,
    GraphAttentionBlock,
    bfs_order,
    degree_of,
    segment_softmax,
)
from rcrn.models.plaque_encoder import (
    ConvNeXtBlock,
    GlobalResponseNorm,
    PlaqueEmbedding,
    PlaqueEncoder,
    TransformerBlock,
)
from rcrn.models.rcrn import RCRNModel, RCRNOutput
from rcrn.models.resistance_head import ResistanceHead, ResistanceOutput
from rcrn.models.route_head import RouteFields, RouteHead, RouteOperator
from rcrn.models.scorer import (
    GatedScorer,
    PlaqueOrdering,
    deferral_share,
    degenerate_orderings,
)
from rcrn.models.state_space import BidirectionalScan, SelectiveStateSpace, StateSpaceStack

__all__ = [
    "COUPLING_KINDS",
    "BidirectionalScan",
    "CompositionHead",
    "CompositionOutput",
    "ConvNeXtBlock",
    "CouplingOutput",
    "DeclineToConditionGate",
    "EdgeConditionedConv",
    "FragmentAttentionPool",
    "GateCalibration",
    "GateDecision",
    "GatedScorer",
    "GlobalResponseNorm",
    "GraphAttentionBlock",
    "InteractionEstimator",
    "InteractionSurfaces",
    "MolMambaBlock",
    "MolMambaEncoder",
    "MolecularEmbedding",
    "PlaqueEmbedding",
    "PlaqueEncoder",
    "PlaqueOrdering",
    "RCRNModel",
    "RCRNOutput",
    "ReactionNormCoupling",
    "ResistanceHead",
    "ResistanceOutput",
    "RouteFields",
    "RouteHead",
    "RouteOperator",
    "SelectiveStateSpace",
    "StateSpaceStack",
    "TransformerBlock",
    "bfs_order",
    "calibrate_threshold",
    "deferral_share",
    "degenerate_orderings",
    "degree_of",
    "false_conditioning_rate",
    "interaction_statistic",
    "perturbation_noise",
    "route_share",
    "segment_softmax",
    "shrinkage_covariance",
]
