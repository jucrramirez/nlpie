from __future__ import annotations

from typing import Optional

from .base import Explanation, ExplanationProvider


class ProjectionExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["projection"]

    def explain(self, report) -> Optional[Explanation]:
        projection = getattr(report, "projection", None)
        if not projection:
            return None

        trust_scores = [p.trustworthiness for p in projection]
        cont_scores = [p.continuity for p in projection]
        mean_trust = sum(trust_scores) / len(trust_scores)
        mean_cont = sum(cont_scores) / len(cont_scores)

        if mean_trust < 0.5 or mean_cont < 0.5:
            severity = "critical"
            summary = f"Projection quality is poor (mean Trustworthiness={mean_trust:.3f}, mean Continuity={mean_cont:.3f})."
            detail = "The low-dimensional projection fails to preserve neighbourhood structure from the original space."
            recommendation = "Try a different dimensionality reduction method (e.g., UMAP instead of t-SNE) or increase the projection dimensionality."
        elif mean_trust < 0.8 or mean_cont < 0.8:
            severity = "warning"
            summary = f"Projection quality is moderate (mean Trustworthiness={mean_trust:.3f}, mean Continuity={mean_cont:.3f})."
            detail = "Some neighbourhood structure is lost in the projection."
            recommendation = "Consider tuning projection parameters (perplexity, n_neighbors) or using a higher-dimensional projection."
        else:
            severity = "info"
            summary = f"Projection quality is good (mean Trustworthiness={mean_trust:.3f}, mean Continuity={mean_cont:.3f})."
            detail = "The low-dimensional projection preserves neighbourhood structure well."
            recommendation = "No action required."

        return Explanation(
            metric="projection",
            severity=severity,
            summary=summary,
            detail=detail,
            recommendation=recommendation,
        )
