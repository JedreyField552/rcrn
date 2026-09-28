"""Clinical-increment statistics.

Ref: Table 4 (incremental area under the curve over the clinical baseline, net
benefit at a 20% threshold, calibration slope and the share of records whose
individual increment reaches the published minimum important difference);
Methods, "Evaluation protocol and statistical analysis" (reclassification and
decision-curve measures report the incremental value).

The area under the curve is the Mann-Whitney statistic with midranks for ties,
written out rather than delegated, because the verification pass recomputes it
from the pairwise definition.
"""

from __future__ import annotations

import numpy as np


def auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve from the pairwise agreement definition."""

    scores = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.shape != targets.shape:
        raise ValueError("predictions and labels must be aligned")
    positives = scores[targets == 1]
    negatives = scores[targets == 0]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("both classes are required")
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (positives.size * negatives.size))


def _placement_values(predictions: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = predictions[labels == 1]
    negatives = predictions[labels == 0]
    case_values = np.array(
        [np.mean(value > negatives) + 0.5 * np.mean(value == negatives) for value in positives],
        dtype=np.float64,
    )
    control_values = np.array(
        [np.mean(positives > value) + 0.5 * np.mean(positives == value) for value in negatives],
        dtype=np.float64,
    )
    return case_values, control_values


def auc_variance(predictions: np.ndarray, labels: np.ndarray) -> float:
    """DeLong variance of the area under the curve."""

    scores = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    case_values, control_values = _placement_values(scores, targets)
    return float(
        np.var(case_values, ddof=1) / case_values.size
        + np.var(control_values, ddof=1) / control_values.size
    )


def auc_covariance(first: np.ndarray, second: np.ndarray, labels: np.ndarray) -> float:
    """DeLong covariance between two scores evaluated on the same records."""

    targets = np.asarray(labels, dtype=np.int64)
    case_first, control_first = _placement_values(np.asarray(first, dtype=np.float64), targets)
    case_second, control_second = _placement_values(np.asarray(second, dtype=np.float64), targets)
    return float(
        np.cov(case_first, case_second, ddof=1)[0, 1] / case_first.size
        + np.cov(control_first, control_second, ddof=1)[0, 1] / control_first.size
    )


def incremental_auc(
    baseline: np.ndarray, augmented: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Difference in area under the curve with its paired standard error."""

    baseline_auc = auc(baseline, labels)
    augmented_auc = auc(augmented, labels)
    variance = (
        auc_variance(baseline, labels)
        + auc_variance(augmented, labels)
        - 2.0 * auc_covariance(baseline, augmented, labels)
    )
    return {
        "baseline_auc": baseline_auc,
        "augmented_auc": augmented_auc,
        "increment": augmented_auc - baseline_auc,
        "standard_error": float(np.sqrt(max(variance, 0.0))),
    }


def net_benefit(predictions: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Decision-curve net benefit at a decision threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("decision threshold must lie in (0, 1)")
    probabilities = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != targets.shape:
        raise ValueError("predictions and labels must be aligned")
    total = probabilities.size
    if total == 0:
        return 0.0
    positive = probabilities >= threshold
    true_positive = float(np.sum(positive & (targets == 1))) / total
    false_positive = float(np.sum(positive & (targets == 0))) / total
    return true_positive - false_positive * threshold / (1.0 - threshold)


def continuous_nri(baseline: np.ndarray, augmented: np.ndarray, labels: np.ndarray) -> float:
    """Pencina continuous net reclassification improvement."""

    base = np.asarray(baseline, dtype=np.float64)
    aug = np.asarray(augmented, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    movement = aug - base
    upward_cases = float(np.mean(movement[targets == 1] > 0))
    downward_cases = float(np.mean(movement[targets == 1] < 0))
    upward_controls = float(np.mean(movement[targets == 0] > 0))
    downward_controls = float(np.mean(movement[targets == 0] < 0))
    return (upward_cases - downward_cases) - (upward_controls - downward_controls)


def mcid_crossing_share(individual_increments: np.ndarray, reference: float) -> float:
    """Share of records whose own increment reaches the published reference level."""

    increments = np.asarray(individual_increments, dtype=np.float64)
    if increments.size == 0:
        return 0.0
    return float(np.mean(increments >= reference))


def weighted_pool(arm_values: np.ndarray, arm_sizes: np.ndarray) -> float:
    """Size-weighted pool of a quantity reported per arm."""

    values = np.asarray(arm_values, dtype=np.float64)
    sizes = np.asarray(arm_sizes, dtype=np.float64)
    if values.shape != sizes.shape:
        raise ValueError("values and sizes must be aligned")
    total = sizes.sum()
    if total == 0.0:
        return 0.0
    return float(np.sum(values * sizes) / total)


def decision_curve_table(
    predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Net benefit evaluated over a threshold grid."""

    return np.array(
        [net_benefit(predictions, labels, float(threshold)) for threshold in thresholds],
        dtype=np.float64,
    )
