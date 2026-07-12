"""High-level helpers for projection quality evaluation (TASK-005).

This module provides convenience functions to compute trustworthiness and
continuity scores for UMAP, t-SNE, or any other dimensionality-reduction
projection, and to display a summary report.

Example usage::

    import numpy as np
    from python.metrics.projection import evaluate_projection

    high = np.random.randn(200, 64)
    low  = np.random.randn(200, 2)   # e.g. from umap-learn
    report = evaluate_projection(high, low, k_values=[5, 10, 20])
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from nlpie._api import MatrixLike, _to_matrix, trustworthiness, continuity
from nlpie._errors import PreprocessingError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionScores:
    """Projection quality scores for a single ``k`` value.

    Attributes:
        k:                 Neighbourhood size.
        trustworthiness:   Trustworthiness score in ``[0, 1]``.
        continuity:        Continuity score in ``[0, 1]``.
    """

    k: int
    trustworthiness: float
    continuity: float

    def __str__(self) -> str:
        return (
            f"k={self.k:>3} | "
            f"Trustworthiness={self.trustworthiness:.4f} | "
            f"Continuity={self.continuity:.4f}"
        )


@dataclass
class ProjectionReport:
    """Aggregated projection quality report across multiple ``k`` values.

    Attributes:
        scores: List of :class:`ProjectionScores`, one per ``k`` value.
    """

    scores: list[ProjectionScores] = field(default_factory=list)

    def __str__(self) -> str:
        header = "Projection Quality Report\n" + "=" * 55
        rows = "\n".join(str(s) for s in self.scores)
        return f"{header}\n{rows}"

    @property
    def mean_trustworthiness(self) -> float:
        """Mean trustworthiness across all ``k`` values."""
        if not self.scores:
            return 0.0
        return sum(s.trustworthiness for s in self.scores) / len(self.scores)

    @property
    def mean_continuity(self) -> float:
        """Mean continuity across all ``k`` values."""
        if not self.scores:
            return 0.0
        return sum(s.continuity for s in self.scores) / len(self.scores)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_projection(
    high_dim: MatrixLike,
    low_dim: MatrixLike,
    k_values: Sequence[int] = (5, 10, 20),
) -> ProjectionReport:
    """Evaluates projection quality at multiple neighbourhood sizes.

    Computes trustworthiness and continuity for each value in ``k_values``
    and returns a :class:`ProjectionReport` summarising the results.

    Args:
        high_dim: Embedding matrix in the original space ``(n_samples, d_high)``.
        low_dim:  Embedding matrix in the projected space ``(n_samples, d_low)``.
        k_values: Sequence of neighbourhood sizes to evaluate (default ``[5, 10, 20]``).

    Returns:
        A :class:`ProjectionReport` containing per-``k`` scores.

    Raises:
        PreprocessingError: If shapes are inconsistent or any ``k`` is invalid.
        ValueError:         If ``k_values`` is empty.
    """
    if not k_values:
        raise ValueError("`k_values` must contain at least one value.")

    high_mat = _to_matrix(high_dim)
    low_mat = _to_matrix(low_dim)

    scores: list[ProjectionScores] = []
    for k in k_values:
        try:
            t = trustworthiness(high_mat, low_mat, k)
            c = continuity(high_mat, low_mat, k)
        except PreprocessingError as exc:
            raise PreprocessingError(
                f"Failed to compute projection metrics for k={k}: {exc}"
            ) from exc
        scores.append(ProjectionScores(k=k, trustworthiness=t, continuity=c))

    return ProjectionReport(scores=scores)
