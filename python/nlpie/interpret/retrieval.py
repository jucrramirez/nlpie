from __future__ import annotations

from ..thresholds import (
    RETRIEVAL_CRITICAL,
    RETRIEVAL_WARNING,
    classify_lower_is_worse,
)
from .base import Explanation, ExplanationProvider


class RetrievalExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["retrieval"]

    def explain(self, report) -> list[Explanation]:
        retrieval = getattr(report, "retrieval", None)
        if not retrieval:
            return []

        max_recall = max(r.recall for r in retrieval)
        max_ndcg = max(r.ndcg for r in retrieval)

        severity = classify_lower_is_worse(max_recall, RETRIEVAL_CRITICAL, RETRIEVAL_WARNING)

        if severity == "critical":
            summary = (
                f"Retrieval quality is very low (max Recall@K={max_recall:.3f}, "
                f"max nDCG={max_ndcg:.3f})."
            )
            detail = "The embedding space struggles to rank relevant items near the top."
            recommendation = (
                "Consider fine-tuning embeddings with a retrieval-aware loss (e.g., "
                "triplet loss, contrastive loss, or ListNet). Check if the embedding "
                "space is aligned with the retrieval task."
            )
        elif severity == "warning":
            summary = (
                f"Retrieval quality is moderate (max Recall@K={max_recall:.3f}, "
                f"max nDCG={max_ndcg:.3f})."
            )
            detail = "Retrieval performance is reasonable but could be improved."
            recommendation = (
                "Consider increasing embedding dimensionality, fine-tuning with a "
                "ranking loss, or adjusting the retrieval strategy."
            )
        else:
            summary = (
                f"Retrieval quality is good (max Recall@K={max_recall:.3f}, "
                f"max nDCG={max_ndcg:.3f})."
            )
            detail = "The embedding space supports effective retrieval."
            recommendation = "No action required."

        return [
            Explanation(
                metric="retrieval",
                severity=severity,
                summary=summary,
                detail=detail,
                recommendation=recommendation,
            )
        ]
