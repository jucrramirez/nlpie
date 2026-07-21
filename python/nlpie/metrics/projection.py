"""High-level helpers for projection quality evaluation (TASK-005).

This module provides convenience functions to compute trustworthiness and
continuity scores for UMAP, t-SNE, or any other dimensionality-reduction
projection, and to display a summary report.

Example usage::

    import numpy as np
    from nlpie.metrics.projection import evaluate_projection

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
from nlpie.metrics.quality import ProjectionMetrics


@dataclass
class ProjectionReport:
    scores: list[ProjectionMetrics] = field(default_factory=list)

    def __str__(self) -> str:
        header = "Projection Quality Report\n" + "=" * 55
        rows = "\n".join(str(s) for s in self.scores)
        return f"{header}\n{rows}"

    @property
    def mean_trustworthiness(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.trustworthiness for s in self.scores) / len(self.scores)

    @property
    def mean_continuity(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.continuity for s in self.scores) / len(self.scores)


def evaluate_projection(
    high_dim: MatrixLike,
    low_dim: MatrixLike,
    k_values: Sequence[int] = (5, 10, 20),
) -> ProjectionReport:
    if not k_values:
        raise ValueError("`k_values` must contain at least one value.")

    high_mat = _to_matrix(high_dim)
    low_mat = _to_matrix(low_dim)

    scores: list[ProjectionMetrics] = []
    for k in k_values:
        try:
            t = trustworthiness(high_mat, low_mat, k)
            c = continuity(high_mat, low_mat, k)
        except PreprocessingError as exc:
            raise PreprocessingError(
                f"Failed to compute projection metrics for k={k}: {exc}"
            ) from exc
        scores.append(ProjectionMetrics(k=k, trustworthiness=t, continuity=c))

    return ProjectionReport(scores=scores)
