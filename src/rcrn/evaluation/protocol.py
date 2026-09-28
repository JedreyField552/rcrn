"""The pre-specified analysis plan.

Ref: Results, Section 2.1; Methods, "Evaluation protocol and statistical analysis"
(one main comparison, a pre-specified success floor, twelve planned tests under
Holm-Bonferroni, an exploratory set controlled at a false discovery rate, and no
adjustment of the thresholds after the evaluation set was touched).

The plan is represented as data so that the primary criterion can be evaluated
against whatever was actually produced, and so that a floor which the design cannot
reach is visible rather than silently missed.
"""

from __future__ import annotations

from dataclasses import dataclass

PRIMARY_METRIC = "rank_agreement"
REFERENCE_LEVEL_SOURCE = "published reference for an incremental imaging layer"


@dataclass(slots=True)
class PlannedComparison:
    """One comparison fixed before the outcome fields were joined."""

    name: str
    stratum: str
    comparator: str
    metric: str
    threshold: float
    family: str

    def met(self, observed: float) -> bool:
        return observed >= self.threshold

    def margin(self, observed: float) -> float:
        return observed - self.threshold

    def as_record(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "stratum": self.stratum,
            "comparator": self.comparator,
            "metric": self.metric,
            "threshold": self.threshold,
            "family": self.family,
        }


@dataclass(slots=True)
class AnalysisPlan:
    """The confirmatory and exploratory families of the study."""

    primary: PlannedComparison
    secondary: tuple[PlannedComparison, ...] = ()
    exploratory: tuple[PlannedComparison, ...] = ()
    confirmatory_family_size: int = 1
    multiplicity_method: str = "holm"
    exploratory_fdr: float = 0.05
    bootstrap_resamples: int = 2000
    seed_count: int = 24
    reference_level: float = 0.07
    scoring_unit: str = "plaque_record"
    adjustments_after_evaluation: int = 0

    def confirmatory_metrics(self) -> dict[str, tuple[str, float]]:
        """Metric name and threshold of every comparison in the confirmatory family."""

        entries = [self.primary, *self.secondary]
        return {entry.name: (entry.metric, entry.threshold) for entry in entries}

    def evaluate(self, observed: dict[str, float]) -> dict[str, object]:
        """Status of the primary criterion and of every secondary comparison."""

        primary_met = self.primary.met(observed.get(self.primary.name, float("-inf")))
        secondary_status = {
            entry.name: entry.met(observed.get(entry.name, float("-inf")))
            for entry in self.secondary
        }
        return {
            "primary_met": primary_met,
            "primary_margin": self.primary.margin(observed.get(self.primary.name, float("-inf"))),
            "secondary": secondary_status,
            "thresholds_adjusted_after_evaluation": self.adjustments_after_evaluation,
        }

    def as_record(self) -> dict[str, object]:
        return {
            "primary": self.primary.as_record(),
            "secondary": [entry.as_record() for entry in self.secondary],
            "exploratory": [entry.as_record() for entry in self.exploratory],
            "confirmatory_family_size": self.confirmatory_family_size,
            "multiplicity_method": self.multiplicity_method,
            "exploratory_fdr": self.exploratory_fdr,
            "bootstrap_resamples": self.bootstrap_resamples,
            "seed_count": self.seed_count,
            "reference_level": self.reference_level,
            "scoring_unit": self.scoring_unit,
            "thresholds_adjusted_after_evaluation": self.adjustments_after_evaluation,
        }


def default_plan(
    reference_stratum: str,
    primary_stratum: str,
    rank_floor: float,
    reference_level: float,
    family_size: int,
    bootstrap_resamples: int,
    seed_count: int,
    exploratory_fdr: float,
) -> AnalysisPlan:
    """Assemble the plan the manuscript describes."""

    primary = PlannedComparison(
        name="primary_rank_agreement",
        stratum=primary_stratum,
        comparator="strongest single-modality baseline",
        metric=PRIMARY_METRIC,
        threshold=rank_floor,
        family="confirmatory",
    )
    secondary = (
        PlannedComparison(
            name="incremental_auc",
            stratum=primary_stratum,
            comparator="clinical variables baseline",
            metric="incremental_auc",
            threshold=reference_level,
            family="confirmatory",
        ),
        PlannedComparison(
            name="reader_agreement",
            stratum=primary_stratum,
            comparator="unassisted reading",
            metric="agreement_improvement",
            threshold=0.10,
            family="confirmatory",
        ),
    )
    return AnalysisPlan(
        primary=primary,
        secondary=secondary,
        exploratory=(),
        confirmatory_family_size=family_size,
        multiplicity_method="holm",
        exploratory_fdr=exploratory_fdr,
        bootstrap_resamples=bootstrap_resamples,
        seed_count=seed_count,
        reference_level=reference_level,
        scoring_unit="plaque_record",
    )


def reachable_margin(threshold: float, comparator: float, ceiling: float = 1.0) -> dict[str, float]:
    """Whether a pre-specified floor is arithmetically reachable on a bounded scale."""

    return {
        "threshold": threshold,
        "comparator": comparator,
        "headroom": ceiling - comparator,
        "reachable": threshold <= ceiling - comparator,
    }


def unadjusted_after_evaluation(plan: AnalysisPlan) -> bool:
    """Confirms the plan records no post-hoc adjustment."""

    return plan.adjustments_after_evaluation == 0


def evidence_footing(primary_met: bool, secondary_met: bool) -> str:
    """Label the evidential footing of a comparison outcome."""

    if primary_met and secondary_met:
        return "confirmatory"
    if primary_met:
        return "primary_only"
    return "not_confirmed"
