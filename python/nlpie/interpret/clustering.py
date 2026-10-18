from __future__ import annotations

from ..thresholds import (
    ARI_CRITICAL,
    ARI_WARNING,
    DAVIES_BOULDIN_CRITICAL,
    DAVIES_BOULDIN_WARNING,
    NMI_WARNING,
    SILHOUETTE_CRITICAL,
    SILHOUETTE_WARNING,
)
from .base import Explanation, ExplanationProvider


class ClusteringExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["clustering"]

    def explain(self, report) -> list[Explanation]:
        clustering = getattr(report, "clustering", None)
        if clustering is None:
            return []

        issues: list[str] = []
        if clustering.ari is not None and clustering.ari < ARI_WARNING:
            issues.append(f"low ARI ({clustering.ari:.3f})")
        if clustering.nmi is not None and clustering.nmi < NMI_WARNING:
            issues.append(f"low NMI ({clustering.nmi:.3f})")
        if clustering.silhouette < SILHOUETTE_WARNING:
            issues.append(f"low Silhouette ({clustering.silhouette:.3f})")
        if clustering.davies_bouldin > DAVIES_BOULDIN_WARNING:
            issues.append(f"high Davies-Bouldin ({clustering.davies_bouldin:.3f})")

        comparisons = (
            f"ARI={clustering.ari:.3f}, NMI={clustering.nmi:.3f}, Purity={clustering.purity:.3f}"
            if clustering.ari is not None
            and clustering.nmi is not None
            and clustering.purity is not None
            else "ARI=N/A, NMI=N/A, Purity=N/A"
        )
        detail = (
            f"{comparisons}, Silhouette (cosine)={clustering.silhouette:.3f}, "
            f"Davies-Bouldin (cosine)={clustering.davies_bouldin:.3f}."
        )

        if not issues:
            return [
                Explanation(
                    metric="clustering",
                    severity="info",
                    summary="Clustering metrics are within acceptable ranges.",
                    detail=detail,
                    recommendation="No action required.",
                )
            ]

        summary = f"Clustering quality issues detected: {', '.join(issues)}."
        severity = (
            "critical"
            if (
                (clustering.ari is not None and clustering.ari < ARI_CRITICAL)
                or clustering.silhouette < SILHOUETTE_CRITICAL
                or clustering.davies_bouldin > DAVIES_BOULDIN_CRITICAL
            )
            else "warning"
        )
        if clustering.silhouette < SILHOUETTE_WARNING:
            detail += " Points are not well-separated by cluster."
        if clustering.davies_bouldin > DAVIES_BOULDIN_WARNING:
            detail += " Cluster overlap is high according to Davies-Bouldin."
        recommendation = (
            "Consider re-embedding with a contrastive or triplet loss. "
            "If labels are noisy, consider label cleaning or soft clustering."
        )

        return [
            Explanation(
                metric="clustering",
                severity=severity,
                summary=summary,
                detail=detail.strip(),
                recommendation=recommendation,
            )
        ]
