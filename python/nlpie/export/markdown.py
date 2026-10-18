from __future__ import annotations

from .base import ReportExporter
from .presentation import build_presentation, metric_views_by_family


def _status_label(status: str) -> str:
    return {
        "critical": "Critical",
        "warning": "Warning",
        "info": "Healthy",
        "measured": "Measured",
        "unavailable": "Not measured",
    }.get(status, status.title())


class MarkdownExporter(ReportExporter):
    def to_string(self, report, interpretation=None) -> str:
        presentation = build_presentation(report, interpretation)
        lines = [
            f"# Embedding Quality Report: {report.model_name}",
            "",
            f"- **Samples:** {report.n_samples}",
            f"- **Dimensions:** {report.n_dims}",
            f"- **Status:** {_status_label(presentation.status)}",
            "",
            "## Executive Summary",
            "",
            f"**{presentation.headline}**",
            "",
            presentation.why_it_matters,
            "",
            "### Primary blockers",
            "",
        ]
        if presentation.primary_blockers:
            lines.extend(f"- {item}" for item in presentation.primary_blockers)
        else:
            lines.append("- None identified in the measured checks.")
        lines.extend(["", "### Healthy signals", ""])
        if presentation.healthy_signals:
            lines.extend(f"- {item}" for item in presentation.healthy_signals)
        else:
            lines.append("- No additional healthy signals were reported.")

        lines.extend(
            [
                "",
                "## Metric Interpretation",
                "",
                "| Metric | Value | Status | Ideal direction | Reference |",
                "|---|---:|---|---|---|",
            ]
        )
        for metric in presentation.metric_interpretation:
            lines.append(
                f"| {metric.metric} | {metric.formatted} | {_status_label(metric.status)} | "
                f"{metric.ideal} | {metric.reference} |"
            )

        lines.extend(["", "## Interpretation Details", ""])
        if presentation.interpretation.explanations:
            for explanation in presentation.interpretation.explanations:
                lines.append(
                    f"- **{explanation.metric}** "
                    f"({_status_label(explanation.severity)}): {explanation.summary}"
                )
                if explanation.detail:
                    lines.append(f"  - {explanation.detail}")
                if explanation.recommendation:
                    lines.append(f"  - Recommendation: {explanation.recommendation}")
        else:
            lines.append("No interpretation findings were produced.")

        lines.extend(["", "## Full Metric Coverage", ""])
        for family, metrics in metric_views_by_family(presentation).items():
            family_title = {
                "Intrinsic geometry": "Intrinsic Metrics",
                "Geometry": "Geometry & Pathology",
            }.get(family, family)
            lines.extend(
                [f"### {family_title}", "", "| Metric | Value | Interpretation |", "|---|---:|---|"]
            )
            for metric in metrics:
                explanation_text = metric.explanation
                if not metric.available:
                    explanation_text = f"{explanation_text} Not measured."
                lines.append(f"| {metric.metric} | {metric.formatted} | {explanation_text} |")
            lines.append("")

        lines.extend(["## Recommended Next Steps", ""])
        if presentation.next_steps:
            lines.extend(
                f"{index}. {step}" for index, step in enumerate(presentation.next_steps, 1)
            )
        else:
            lines.append("No corrective action is required from the measured checks.")

        lines.extend(
            [
                "",
                "## Evidence Availability",
                "",
                "| Evidence | Available |",
                "|---|---|",
            ]
        )
        for key, available in presentation.data_availability.items():
            lines.append(f"| {key} | {'Yes' if available else 'No — not measured'} |")

        lines.extend(
            [
                "",
                "## How to Read This Report",
                "",
                "- **Intrinsic geometry:** describes how similar vectors are to one another "
                "and whether the space is overly aligned.",
                "- **Clustering:** describes separation and overlap between groups; "
                "Silhouette is higher-is-better and Davies–Bouldin is lower-is-better.",
                "- **Geometry:** describes usable dimensionality and whether a few vectors "
                "dominate neighbourhoods.",
                "- **Projection:** describes how faithfully a lower-dimensional view "
                "preserves neighbourhood relationships.",
                "- **Retrieval:** describes whether relevant items are found and ranked near "
                "the top; higher values are better and 100% is the maximum.",
                "- **Not measured:** means the required input or raw trace was not present; "
                "it is not a failing score.",
            ]
        )
        return "\n".join(lines)

    def export(self, report, path: str, interpretation=None) -> None:
        with open(path, "w") as f:
            f.write(self.to_string(report, interpretation=interpretation))
