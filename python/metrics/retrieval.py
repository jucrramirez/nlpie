"""High-level helpers for retrieval and ranking metric evaluation (TASK-006).

This module provides convenience functions to compute retrieval quality metrics
(Recall@K, Precision@K, MRR, nDCG@K, Coverage@K) across multiple queries and
cut-off values, and to produce a structured report.

Example usage::

    from python.metrics.retrieval import evaluate_retrieval

    all_retrieved = [[0, 2, 5], [1, 3, 6]]
    all_relevant  = [[0, 2],    [1]]
    report = evaluate_retrieval(all_retrieved, all_relevant, k_values=[1, 2, 3])
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from nlpie._api import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    coverage_at_k,
)
from nlpie._errors import PreprocessingError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalScores:
    """Aggregated retrieval metric scores for a single cut-off ``k``.

    Attributes:
        k:            Cut-off rank.
        recall:       Mean Recall\\@K across all queries.
        precision:    Mean Precision\\@K across all queries.
        mrr:          Mean Reciprocal Rank (query-level average when all
                      queries are scored at the same retrieval list length).
        ndcg:         Mean nDCG\\@K across all queries.
        coverage:     Coverage\\@K across all queries.
    """

    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    coverage: float

    def __str__(self) -> str:
        return (
            f"k={self.k:>3} | "
            f"R@K={self.recall:.4f}  "
            f"P@K={self.precision:.4f}  "
            f"MRR={self.mrr:.4f}  "
            f"nDCG={self.ndcg:.4f}  "
            f"Cov={self.coverage:.4f}"
        )


@dataclass
class RetrievalReport:
    """Aggregated retrieval quality report across multiple cut-off ``k`` values.

    Attributes:
        scores: List of :class:`RetrievalScores`, one per ``k`` value.
    """

    scores: list[RetrievalScores] = field(default_factory=list)

    def __str__(self) -> str:
        header = (
            "Retrieval Quality Report\n"
            + "=" * 72 + "\n"
            + f"{'k':>3}   {'R@K':>7}  {'P@K':>7}  {'MRR':>7}  {'nDCG':>7}  {'Cov':>7}\n"
            + "-" * 72
        )
        rows = "\n".join(str(s) for s in self.scores)
        return f"{header}\n{rows}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Returns the arithmetic mean of ``values``, or ``0.0`` if empty."""
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    all_retrieved: Sequence[Sequence[int]],
    all_relevant: Sequence[Sequence[int]],
    k_values: Sequence[int] = (1, 5, 10),
) -> RetrievalReport:
    """Evaluates retrieval quality across multiple queries and cut-off values.

    Computes Recall\\@K, Precision\\@K, MRR, nDCG\\@K, and Coverage\\@K for each
    value in ``k_values`` and returns a :class:`RetrievalReport`.

    MRR is computed per-query using the full ``retrieved`` list (not capped at
    ``k``), then averaged over queries — consistent with standard IR practice.

    Args:
        all_retrieved: One ranked list of document IDs per query.
        all_relevant:  One ground-truth relevant-ID list per query.
        k_values:      Cut-off ranks to evaluate (default ``[1, 5, 10]``).

    Returns:
        A :class:`RetrievalReport` with one :class:`RetrievalScores` per ``k``.

    Raises:
        PreprocessingError: If lists are mismatched or any ``k`` is invalid.
        ValueError:         If ``all_retrieved`` and ``all_relevant`` differ in
                            length, or ``k_values`` is empty.
    """
    if not k_values:
        raise ValueError("`k_values` must contain at least one value.")
    if len(all_retrieved) != len(all_relevant):
        raise ValueError(
            "`all_retrieved` and `all_relevant` must have the same length."
        )

    # Normalise to plain Python lists once.
    retrieved_lists: list[list[int]] = [list(r) for r in all_retrieved]
    relevant_lists: list[list[int]] = [list(r) for r in all_relevant]
    n_queries = len(retrieved_lists)

    # Compute MRR once (not k-dependent).
    mrr_values: list[float] = []
    for i in range(n_queries):
        try:
            mrr_values.append(
                mean_reciprocal_rank(retrieved_lists[i], relevant_lists[i])
            )
        except PreprocessingError as exc:
            raise PreprocessingError(
                f"Failed to compute MRR for query {i}: {exc}"
            ) from exc

    mean_mrr = _mean(mrr_values)

    scores: list[RetrievalScores] = []
    for k in k_values:
        recall_values: list[float] = []
        precision_values: list[float] = []
        ndcg_values: list[float] = []

        for i in range(n_queries):
            try:
                recall_values.append(recall_at_k(retrieved_lists[i], relevant_lists[i], k))
                precision_values.append(precision_at_k(retrieved_lists[i], relevant_lists[i], k))
                ndcg_values.append(ndcg_at_k(retrieved_lists[i], relevant_lists[i], k))
            except PreprocessingError as exc:
                raise PreprocessingError(
                    f"Failed to compute metrics for query {i}, k={k}: {exc}"
                ) from exc

        try:
            cov = coverage_at_k(retrieved_lists, relevant_lists, k)
        except PreprocessingError as exc:
            raise PreprocessingError(
                f"Failed to compute coverage for k={k}: {exc}"
            ) from exc

        scores.append(
            RetrievalScores(
                k=k,
                recall=_mean(recall_values),
                precision=_mean(precision_values),
                mrr=mean_mrr,
                ndcg=_mean(ndcg_values),
                coverage=cov,
            )
        )

    return RetrievalReport(scores=scores)
