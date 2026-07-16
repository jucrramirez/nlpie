from __future__ import annotations

from typing import Optional

from .base import Explanation, ExplanationProvider


class RetrievalExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["retrieval"]

    def explain(self, report) -> Optional[Explanation]:
        retrieval = getattr(report, "retrieval", None)
        if not retrieval:
            return None

        max_recall = max(r.recall for r in retrieval)
        max_ndcg = max(r.ndcg for r in retrieval)

        if max_recall < 0.3:
            severity = "critical"
            summary = f"Retrieval quality is very low (max Recall@K={max_recall:.3f}, max nDCG={max_ndcg:.3f})."
            detail = "The embedding space struggles to rank relevant items near the top."
            recommendation = "Consider fine-tuning embeddings with a retrieval-aware loss (e.g., triplet loss, contrastive loss, or ListNet). Check if the embedding space is aligned with the retrieval task."
        elif max_recall < 0.7:
            severity = "warning"
            summary = f"Retrieval quality is moderate (max Recall@K={max_recall:.3f}, max nDCG={max_ndcg:.3f})."
            detail = "Retrieval performance is reasonable but could be improved."
            recommendation = "Consider increasing embedding dimensionality, fine-tuning with a ranking loss, or adjusting the retrieval strategy."
        else:
            severity = "info"
            summary = f"Retrieval quality is good (max Recall@K={max_recall:.3f}, max nDCG={max_ndcg:.3f})."
            detail = "The embedding space supports effective retrieval."
            recommendation = "No action required."

        return Explanation(
            metric="retrieval",
            severity=severity,
            summary=summary,
            detail=detail,
            recommendation=recommendation,
        )
