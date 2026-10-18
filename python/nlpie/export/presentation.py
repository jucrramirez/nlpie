"""Shared, data-driven presentation data for quality-report exporters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..interpret.base import InterpretationReport
from ..interpret.registry import ExplanationRegistry
from ..interpret.story import StoryData, build_story
from ..thresholds import (
    ARI_CRITICAL,
    ARI_WARNING,
    DAVIES_BOULDIN_CRITICAL,
    DAVIES_BOULDIN_WARNING,
    EFFECTIVE_RANK_CRITICAL_FRAC,
    EFFECTIVE_RANK_WARNING_FRAC,
    HUBNESS_CRITICAL,
    HUBNESS_WARNING,
    INFO,
    MEAN_SIMILARITY_CRITICAL,
    MEAN_SIMILARITY_WARNING,
    PROJECTION_CRITICAL,
    PROJECTION_WARNING,
    RETRIEVAL_CRITICAL,
    RETRIEVAL_WARNING,
    SILHOUETTE_CRITICAL,
    SILHOUETTE_WARNING,
    SIMILARITY_STD_WARNING,
    WARNING,
    classify_higher_is_worse,
    classify_lower_is_worse,
    worst_severity,
)


@dataclass(frozen=True)
class MetricView:
    """A renderer-neutral description of one displayed metric."""

    metric: str
    family: str
    value: Any
    formatted: str
    available: bool
    status: str
    ideal: str
    reference: str
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "family": self.family,
            "value": self.value,
            "formatted": self.formatted,
            "available": self.available,
            "status": self.status,
            "ideal": self.ideal,
            "reference": self.reference,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ReportPresentation:
    """Structured report content shared by HTML, Markdown, and JSON."""

    story: StoryData
    status: str
    headline: str
    why_it_matters: str
    primary_blockers: list[str]
    healthy_signals: list[str]
    metric_interpretation: list[MetricView]
    next_steps: list[str]
    data_availability: dict[str, Any]

    @property
    def interpretation(self) -> InterpretationReport:
        return InterpretationReport(
            explanations=[exp for group in self.story.groups for exp in group.explanations]
        )


def resolve_interpretation(report, interpretation=None) -> InterpretationReport:
    """Return an explicit interpretation or derive one from ``report``."""
    if interpretation is not None:
        return interpretation
    return ExplanationRegistry.explain_all(report)


def _explanation_by_metric(story: StoryData) -> dict[str, Any]:
    return {exp.metric: exp for group in story.groups for exp in group.explanations}


def _status_for_exp(explanations: dict[str, Any], metric: str) -> str:
    explanation = explanations.get(metric)
    return explanation.severity if explanation is not None else "measured"


def _metric(
    metric: str,
    family: str,
    value: Any,
    formatted: str,
    status: str,
    ideal: str,
    reference: str,
    explanation: str,
) -> MetricView:
    return MetricView(
        metric=metric,
        family=family,
        value=value,
        formatted=formatted,
        available=value is not None,
        status=status if value is not None else "unavailable",
        ideal=ideal,
        reference=reference,
        explanation=explanation,
    )


def _build_metric_views(report) -> list[MetricView]:
    views: list[MetricView] = []
    intrinsic = getattr(report, "intrinsic", None)
    if intrinsic is not None:
        views.extend(
            [
                _metric(
                    "intrinsic.mean",
                    "Intrinsic geometry",
                    intrinsic.mean,
                    f"{intrinsic.mean:.4f}",
                    classify_higher_is_worse(
                        intrinsic.mean, MEAN_SIMILARITY_CRITICAL, MEAN_SIMILARITY_WARNING
                    ),
                    "Lower alignment; values near 0 are typically desirable.",
                    (
                        f"Warning > {MEAN_SIMILARITY_WARNING:.2f}; "
                        f"critical > {MEAN_SIMILARITY_CRITICAL:.2f}"
                    ),
                    "Average cosine similarity between pairs of vectors.",
                ),
                _metric(
                    "intrinsic.std",
                    "Intrinsic geometry",
                    intrinsic.std,
                    f"{intrinsic.std:.4f}",
                    WARNING if intrinsic.std < SIMILARITY_STD_WARNING else INFO,
                    "A sufficiently varied distribution; no universal target.",
                    f"Warning < {SIMILARITY_STD_WARNING:.2f}",
                    "Spread of pairwise similarities; a very narrow spread can indicate collapse.",
                ),
                _metric(
                    "intrinsic.min",
                    "Intrinsic geometry",
                    intrinsic.min,
                    f"{intrinsic.min:.4f}",
                    "measured",
                    "Descriptive range; interpret with mean and standard deviation.",
                    "Descriptive only",
                    "Lowest observed pairwise cosine similarity.",
                ),
                _metric(
                    "intrinsic.max",
                    "Intrinsic geometry",
                    intrinsic.max,
                    f"{intrinsic.max:.4f}",
                    "measured",
                    "Descriptive range; interpret with mean and standard deviation.",
                    "Descriptive only",
                    "Highest observed pairwise cosine similarity.",
                ),
            ]
        )

    clustering = getattr(report, "clustering", None)
    if clustering is not None:
        clustering_values = (
            ("clustering.ari", "ARI", clustering.ari, ARI_CRITICAL, ARI_WARNING),
            ("clustering.nmi", "NMI", clustering.nmi, None, 0.5),
            ("clustering.purity", "Purity", clustering.purity, None, None),
        )
        for key, label, value, critical, warning in clustering_values:
            if value is None:
                views.append(
                    _metric(
                        key,
                        "Clustering",
                        None,
                        "N/A — not measured",
                        "unavailable",
                        "Higher is better; maximum 1.0.",
                        "Requires labels and predictions",
                        f"{label} needs label data and is not available in this report.",
                    )
                )
            else:
                status = (
                    classify_lower_is_worse(value, critical, warning)
                    if critical is not None and warning is not None
                    else (WARNING if warning is not None and value < warning else INFO)
                )
                views.append(
                    _metric(
                        key,
                        "Clustering",
                        value,
                        f"{value:.4f}",
                        status,
                        "Higher is better; maximum 1.0.",
                        "Higher is better",
                        f"{label} compares predicted grouping quality with the reference labels.",
                    )
                )
        views.extend(
            [
                _metric(
                    "clustering.silhouette",
                    "Clustering",
                    clustering.silhouette,
                    f"{clustering.silhouette:.4f}",
                    classify_lower_is_worse(
                        clustering.silhouette, SILHOUETTE_CRITICAL, SILHOUETTE_WARNING
                    ),
                    "Higher is better; maximum 1.0.",
                    f"Warning < {SILHOUETTE_WARNING:.2f}; critical < {SILHOUETTE_CRITICAL:.2f}",
                    "Measures how separated and internally coherent clusters are.",
                ),
                _metric(
                    "clustering.davies_bouldin",
                    "Clustering",
                    clustering.davies_bouldin,
                    f"{clustering.davies_bouldin:.4f}",
                    classify_higher_is_worse(
                        clustering.davies_bouldin,
                        DAVIES_BOULDIN_CRITICAL,
                        DAVIES_BOULDIN_WARNING,
                    ),
                    "Lower is better; 0 is ideal.",
                    (
                        f"Warning > {DAVIES_BOULDIN_WARNING:.2f}; "
                        f"critical > {DAVIES_BOULDIN_CRITICAL:.2f}"
                    ),
                    "Measures cluster overlap; lower values mean cleaner separation.",
                ),
            ]
        )

    geometry = getattr(report, "geometry", None)
    if geometry is not None:
        rank_critical = report.n_dims * EFFECTIVE_RANK_CRITICAL_FRAC
        rank_warning = report.n_dims * EFFECTIVE_RANK_WARNING_FRAC
        views.extend(
            [
                _metric(
                    "geometry.effective_rank",
                    "Geometry",
                    geometry.effective_rank,
                    f"{geometry.effective_rank:.2f} / {report.n_dims}",
                    classify_lower_is_worse(geometry.effective_rank, rank_critical, rank_warning),
                    "Closer to ambient dimensions is better.",
                    f"Warning < {rank_warning:.1f}; critical < {rank_critical:.1f}",
                    "Estimates how many dimensions carry meaningful variation.",
                ),
                _metric(
                    "geometry.mean_similarity",
                    "Geometry",
                    geometry.mean_similarity,
                    f"{geometry.mean_similarity:.4f}",
                    classify_higher_is_worse(
                        geometry.mean_similarity,
                        MEAN_SIMILARITY_CRITICAL,
                        MEAN_SIMILARITY_WARNING,
                    ),
                    "Lower alignment with the global mean is generally better.",
                    (
                        f"Warning > {MEAN_SIMILARITY_WARNING:.2f}; "
                        f"critical > {MEAN_SIMILARITY_CRITICAL:.2f}"
                    ),
                    "Measures whether vectors point toward one dominant global direction.",
                ),
                _metric(
                    "geometry.hubness_skewness",
                    "Geometry",
                    geometry.hubness_skewness,
                    f"{geometry.hubness_skewness:.4f}",
                    classify_higher_is_worse(
                        geometry.hubness_skewness, HUBNESS_CRITICAL, HUBNESS_WARNING
                    ),
                    "Near 0 is better.",
                    f"Warning > {HUBNESS_WARNING:.2f}; critical > {HUBNESS_CRITICAL:.2f}",
                    "Measures whether a small number of points appear in too many neighbour lists.",
                ),
                _metric(
                    "geometry.hubness_k",
                    "Geometry",
                    geometry.hubness_k,
                    str(geometry.hubness_k),
                    "measured",
                    "Context for the hubness calculation.",
                    "Configuration value",
                    "Neighbourhood size used to measure hubness.",
                ),
            ]
        )

    for projection in getattr(report, "projection", []) or []:
        views.extend(
            [
                _metric(
                    f"projection.trustworthiness@{projection.k}",
                    "Projection",
                    projection.trustworthiness,
                    f"{projection.trustworthiness:.4f}",
                    classify_lower_is_worse(
                        projection.trustworthiness, PROJECTION_CRITICAL, PROJECTION_WARNING
                    ),
                    "Closer to 1.0 is better.",
                    f"Warning < {PROJECTION_WARNING:.2f}; critical < {PROJECTION_CRITICAL:.2f}",
                    (
                        "Measures whether nearby high-dimensional neighbours stay nearby "
                        "in the projection."
                    ),
                ),
                _metric(
                    f"projection.continuity@{projection.k}",
                    "Projection",
                    projection.continuity,
                    f"{projection.continuity:.4f}",
                    classify_lower_is_worse(
                        projection.continuity, PROJECTION_CRITICAL, PROJECTION_WARNING
                    ),
                    "Closer to 1.0 is better.",
                    f"Warning < {PROJECTION_WARNING:.2f}; critical < {PROJECTION_CRITICAL:.2f}",
                    (
                        "Measures whether projection neighbours reflect relationships "
                        "in the original space."
                    ),
                ),
            ]
        )

    for retrieval in getattr(report, "retrieval", []) or []:
        for key, label, value, explanation in (
            (
                "recall",
                "Recall",
                retrieval.recall,
                "Share of relevant items retrieved in the top K.",
            ),
            (
                "precision",
                "Precision",
                retrieval.precision,
                "Share of the top K results that are relevant.",
            ),
            (
                "mrr",
                "MRR",
                retrieval.mrr,
                "Rewards putting the first relevant result near the top.",
            ),
            (
                "ndcg",
                "nDCG",
                retrieval.ndcg,
                "Rewards ranking relevant results early, with graded relevance support.",
            ),
            (
                "coverage",
                "Coverage",
                retrieval.coverage,
                "Share of queries for which at least one relevant item was found.",
            ),
        ):
            status = "measured"
            if key in {"recall", "ndcg"}:
                status = classify_lower_is_worse(value, RETRIEVAL_CRITICAL, RETRIEVAL_WARNING)
            views.append(
                _metric(
                    f"retrieval.{key}@{retrieval.k}",
                    "Retrieval",
                    value,
                    f"{value:.1%}",
                    status,
                    "Higher is better; maximum 100%.",
                    (
                        f"Warning < {RETRIEVAL_WARNING:.0%}; critical < {RETRIEVAL_CRITICAL:.0%}"
                        if key in {"recall", "ndcg"}
                        else "Higher is better; no dedicated severity threshold"
                    ),
                    explanation,
                )
            )

    return views


def _headline(story: StoryData, explanations: dict[str, Any]) -> str:
    metrics = set(explanations)
    if story.is_healthy:
        return "Embedding quality is healthy across measured checks."
    if "retrieval" in metrics and story.headline_severity == "critical":
        return "Strong structure, weak retrieval."
    if "hubness" in metrics or "effective_rank" in metrics:
        return "Embedding geometry needs investigation."
    if "projection" in metrics:
        return "Projection fidelity needs investigation."
    return "Embedding quality requires attention."


def _why_it_matters(explanations: dict[str, Any]) -> str:
    if "retrieval" in explanations:
        return (
            "An embedding can look separated or well-conditioned while still placing "
            "task-relevant items too low in the nearest-neighbour ranking."
        )
    return (
        "The readout combines complementary checks because no single metric describes "
        "embedding quality by itself."
    )


def _dedupe_actions(story: StoryData) -> list[str]:
    actions: list[str] = []
    for recommendation in story.recommendations:
        normalized = recommendation.strip().lower().rstrip(".")
        if not normalized or normalized == "no action required":
            continue
        if recommendation not in actions:
            actions.append(recommendation)
    return actions


def build_presentation(report, interpretation=None) -> ReportPresentation:
    """Build all non-visual report content from one report and interpretation."""
    resolved = resolve_interpretation(report, interpretation)
    story = build_story(resolved.explanations)
    by_metric = _explanation_by_metric(story)
    status = "healthy" if story.is_healthy else story.headline_severity
    primary_blockers = [exp.summary for exp in resolved.critical + resolved.warnings]
    healthy_signals = [exp.summary for exp in resolved.info]
    return ReportPresentation(
        story=story,
        status=status,
        headline=_headline(story, by_metric),
        why_it_matters=_why_it_matters(by_metric),
        primary_blockers=primary_blockers,
        healthy_signals=healthy_signals,
        metric_interpretation=_build_metric_views(report),
        next_steps=_dedupe_actions(story),
        data_availability={
            "intrinsic": getattr(report, "intrinsic", None) is not None,
            "clustering": getattr(report, "clustering", None) is not None,
            "geometry": getattr(report, "geometry", None) is not None,
            "projection": bool(getattr(report, "projection", None)),
            "retrieval": bool(getattr(report, "retrieval", None)),
            "pairwise_similarities": bool(getattr(report, "pairwise_similarities", None)),
            "hubness_counts": bool(getattr(report, "hubness_counts", None)),
            "sim_to_mean": bool(getattr(report, "sim_to_mean", None)),
        },
    )


def metric_views_by_family(presentation: ReportPresentation) -> dict[str, list[MetricView]]:
    grouped: dict[str, list[MetricView]] = {}
    for view in presentation.metric_interpretation:
        grouped.setdefault(view.family, []).append(view)
    return grouped


def worst_status(presentation: ReportPresentation, family: str) -> str:
    statuses = [
        view.status
        for view in presentation.metric_interpretation
        if view.family == family and view.status in {"critical", "warning", INFO}
    ]
    return worst_severity(statuses) if statuses else "measured"
