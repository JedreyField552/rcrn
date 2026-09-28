"""Calibration statistics.

Ref: Table 2 Tier 1 (removing the gate moves the calibration slope); Table 4
(calibration slope per arm, site, region and comparator); Discussion (the gate
reduces the slope without changing the discrimination level).

The slope is the coefficient of a one-variable logistic regression of the outcome
on the score, fitted by Newton iterations with a ridge floor so a perfectly
separated score does not diverge.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logistic_fit(
    predictors: np.ndarray, labels: np.ndarray, ridge: float = 1e-6, iterations: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-penalised logistic regression with an intercept, by Newton steps."""

    design = np.column_stack([np.ones(predictors.shape[0]), predictors])
    targets = np.asarray(labels, dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    hessian = np.eye(design.shape[1], dtype=np.float64)
    for _ in range(iterations):
        linear = design @ coefficients
        probabilities = _sigmoid(linear)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        gradient = design.T @ (targets - probabilities) - penalty @ coefficients
        hessian = design.T @ (design * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        coefficients = coefficients + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    standard_errors = np.sqrt(np.clip(np.diag(np.linalg.pinv(hessian)), 0.0, None))
    return coefficients, standard_errors


def calibration_slope(logits: np.ndarray, labels: np.ndarray) -> float:
    """Slope of the calibration regression of the outcome on the score."""

    coefficients, _ = logistic_fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), labels)
    return float(coefficients[1])


def calibration_intercept(logits: np.ndarray, labels: np.ndarray) -> float:
    """Intercept of the calibration regression."""

    coefficients, _ = logistic_fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), labels)
    return float(coefficients[0])


def calibration_in_the_large(logits: np.ndarray, labels: np.ndarray) -> float:
    """Difference between the mean predicted risk and the observed rate."""

    probabilities = _sigmoid(np.asarray(logits, dtype=np.float64))
    targets = np.asarray(labels, dtype=np.float64)
    return float(probabilities.mean() - targets.mean())


def calibration_ratio(logits: np.ndarray, labels: np.ndarray) -> float:
    """Brier-style ratio of the observed to the expected squared error."""

    probabilities = _sigmoid(np.asarray(logits, dtype=np.float64))
    targets = np.asarray(labels, dtype=np.float64)
    observed = float(np.mean((targets - probabilities) ** 2))
    expected = float(np.mean(probabilities * (1.0 - probabilities)))
    if expected == 0.0:
        return 0.0
    return observed / expected
