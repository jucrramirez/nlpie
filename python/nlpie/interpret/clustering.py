from __future__ import annotations

from typing import Optional

from .base import Explanation, ExplanationProvider


class ClusteringExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["clustering"]

    def explain(self, report) -> Optional[Explanation]:
        clustering = getattr(report, "clustering", None)
        if clustering is None:
            return None

        issues: list[str] = []
        if clustering.ari < 0.5:
            issues.append(f"low ARI ({clustering.ari:.3f})")
        if clustering.nmi < 0.5:
            issues.append(f"low NMI ({clustering.nmi:.3f})")
        if clustering.silhouette < 0.25:
            issues.append(f"low Silhouette ({clustering.silhouette:.3f})")

        if not issues:
            return Explanation(
                metric="clustering",
                severity="info",
                summary="Clustering metrics are within acceptable ranges.",
                detail=f"ARI={clustering.ari:.3f}, NMI={clustering.nmi:.3f}, Silhouette={clustering.silhouette:.3f}, CH={clustering.calinski_harabasz:.1f}",
                recommendation="No action required.",
            )

        summary = f"Clustering quality issues detected: {', '.join(issues)}."
        severity = "critical" if clustering.ari < 0.25 or clustering.silhouette < 0.1 else "warning"
        detail = (
            f"ARI={clustering.ari:.3f}, NMI={clustering.nmi:.3f}, "
            f"Silhouette={clustering.silhouette:.3f}, CH={clustering.calinski_harabasz:.1f}. "
        )
        if clustering.silhouette < 0.25:
            detail += "Points are not well-separated by cluster. "
        recommendation = (
            "Consider re-embedding with a contrastive or triplet loss. "
            "If labels are noisy, consider label cleaning or soft clustering."
        )

        return Explanation(
            metric="clustering",
            severity=severity,
            summary=summary,
            detail=detail.strip(),
            recommendation=recommendation,
        )
