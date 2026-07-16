from __future__ import annotations

from .base import ReportExporter


class MarkdownExporter(ReportExporter):
    def to_string(self, report, interpretation=None) -> str:
        lines = [
            f"# Embedding Quality Report: {report.model_name}",
            "",
            f"- **Samples:** {report.n_samples}",
            f"- **Dimensions:** {report.n_dims}",
            "",
        ]

        if hasattr(report, "intrinsic") and report.intrinsic is not None:
            m = report.intrinsic
            lines.extend([
                "## Intrinsic Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean | {m.mean:.4f} |",
                f"| Std | {m.std:.4f} |",
                f"| Min | {m.min:.4f} |",
                f"| Max | {m.max:.4f} |",
                "",
            ])

        if hasattr(report, "clustering") and report.clustering is not None:
            m = report.clustering
            lines.extend([
                "## Clustering Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| ARI | {m.ari:.4f} |",
                f"| NMI | {m.nmi:.4f} |",
                f"| Purity | {m.purity:.4f} |",
                f"| Silhouette | {m.silhouette:.4f} |",
                f"| Calinski-Harabasz | {m.calinski_harabasz:.2f} |",
                "",
            ])

        if hasattr(report, "geometry") and report.geometry is not None:
            m = report.geometry
            lines.extend([
                "## Geometry & Pathology",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Effective Rank | {m.effective_rank:.2f} |",
                f"| Mean Sim-to-Mean | {m.mean_similarity:.4f} |",
                f"| Hubness Skewness | {m.hubness_skewness:.4f} |",
                f"| Hubness K | {m.hubness_k} |",
                "",
            ])

        if hasattr(report, "projection") and report.projection:
            lines.extend([
                "## Projection Quality",
                "",
                "| k | Trustworthiness | Continuity |",
                "|---|----------------|------------|",
            ])
            for p in report.projection:
                lines.append(
                    f"| {p.k} | {p.trustworthiness:.4f} | {p.continuity:.4f} |"
                )
            lines.append("")

        if hasattr(report, "retrieval") and report.retrieval:
            lines.extend([
                "## Retrieval Quality",
                "",
                "| k | R@K | P@K | MRR | nDCG | Coverage |",
                "|---|-----|-----|-----|------|----------|",
            ])
            for r in report.retrieval:
                lines.append(
                    f"| {r.k} | {r.recall:.4f} | {r.precision:.4f} | "
                    f"{r.mrr:.4f} | {r.ndcg:.4f} | {r.coverage:.4f} |"
                )
            lines.append("")

        if interpretation is not None:
            explanations = getattr(interpretation, "explanations", None) or []
            if explanations:
                lines.extend(["", "## Interpretation", ""])
                for exp in explanations:
                    lines.append(f"- **{exp.metric}** ({exp.severity}): {exp.summary}")
                    rec = getattr(exp, "recommendation", "") or ""
                    if rec:
                        lines.append(f"  - *Recommendation:* {rec}")

        return "\n".join(lines)

    def export(self, report, path: str, interpretation=None) -> None:
        with open(path, "w") as f:
            f.write(self.to_string(report, interpretation=interpretation))
