from __future__ import annotations

from ..thresholds import (
    ARI_CRITICAL,
    ARI_WARNING,
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
        if clustering.ari < ARI_WARNING:
            issues.append(f"low ARI ({clustering.ari:.3f})")
        if clustering.nmi < NMI_WARNING:
            issues.append(f"low NMI ({clustering.nmi:.3f})")
        if clustering.silhouette < SILHOUETTE_WARNING:
            issues.append(f"low Silhouette ({clustering.silhouette:.3f})")

        if not issues:
            return [
                Explanation(
                    metric="clustering",
                    severity="info",
                    summary="Clustering metrics are within acceptable ranges.",
                    detail=(
                        f"ARI={clustering.ari:.3f}, NMI={clustering.nmi:.3f}, "
                        f"Silhouette={clustering.silhouette:.3f}, "
                        f"CH={clustering.calinski_harabasz:.1f}"
                    ),
                    recommendation="No action required.",
                )
            ]

        summary = f"Clustering quality issues detected: {', '.join(issues)}."
        severity = (
            "critical"
            if clustering.ari < ARI_CRITICAL or clustering.silhouette < SILHOUETTE_CRITICAL
            else "warning"
        )
        detail = (
            f"ARI={clustering.ari:.3f}, NMI={clustering.nmi:.3f}, "
            f"Silhouette={clustering.silhouette:.3f}, CH={clustering.calinski_harabasz:.1f}. "
        )
        if clustering.silhouette < SILHOUETTE_WARNING:
            detail += "Points are not well-separated by cluster. "
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
