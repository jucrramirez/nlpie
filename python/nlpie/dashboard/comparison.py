from __future__ import annotations

from typing import Optional, Sequence

from nlpie._types import MatrixLike
from nlpie.metrics.quality import (
    EmbeddingQualityReport,
    evaluate_embedding_quality,
)
from ..backends import PlotBackend, PlotlyBackend


def _resolve_backend(backend: Optional[PlotBackend] = None) -> PlotBackend:
    if backend is None:
        return PlotlyBackend()
    return backend


def _collect_metric_values(
    reports: list[EmbeddingQualityReport],
) -> tuple[list[str], list[str], list[list[float]]]:
    model_names = [r.model_name for r in reports]
    metric_names: list[str] = []
    values: list[list[float]] = [[] for _ in reports]

    for idx, r in enumerate(reports):
        if r.intrinsic is not None:
            if "Intrinsic-Mean" not in metric_names:
                metric_names.append("Intrinsic-Mean")
            values[idx].append(r.intrinsic.mean)
        if r.clustering is not None:
            if "ARI" not in metric_names:
                metric_names.extend(["ARI", "NMI", "Silhouette"])
            values[idx].extend([r.clustering.ari, r.clustering.nmi, r.clustering.silhouette])
        if r.geometry is not None:
            if "Eff-Rank" not in metric_names:
                metric_names.append("Eff-Rank")
            values[idx].append(r.geometry.effective_rank)
        if r.projection:
            if "Proj-Trust" not in metric_names:
                metric_names.extend(["Proj-Trust", "Proj-Cont"])
            mean_t = sum(p.trustworthiness for p in r.projection) / len(r.projection)
            mean_c = sum(p.continuity for p in r.projection) / len(r.projection)
            values[idx].extend([mean_t, mean_c])
        if r.retrieval:
            if "R@K" not in metric_names:
                metric_names.extend(["R@K", "nDCG"])
            mean_recall = sum(rr.recall for rr in r.retrieval) / len(r.retrieval)
            mean_ndcg = sum(rr.ndcg for rr in r.retrieval) / len(r.retrieval)
            values[idx].extend([mean_recall, mean_ndcg])

    return model_names, metric_names, values


def compare_and_plot_radar(
    models: dict[str, MatrixLike],
    *,
    labels: Optional[Sequence[int]] = None,
    hubness_k: int = 5,
    backend: Optional[PlotBackend] = None,
    **kwargs,
) -> object:
    reports = []
    for name, emb in models.items():
        r, _ = evaluate_embedding_quality(
            emb, labels=labels, hubness_k=hubness_k, model_name=name, **kwargs
        )
        reports.append(r)
    model_names, metric_names, values = _collect_metric_values(reports)
    b = _resolve_backend(backend)
    return b.comparison_radar(model_names, metric_names, values)


def compare_and_plot_grouped_bar(
    models: dict[str, MatrixLike],
    *,
    labels: Optional[Sequence[int]] = None,
    hubness_k: int = 5,
    backend: Optional[PlotBackend] = None,
    **kwargs,
) -> object:
    reports = []
    for name, emb in models.items():
        r, _ = evaluate_embedding_quality(
            emb, labels=labels, hubness_k=hubness_k, model_name=name, **kwargs
        )
        reports.append(r)
    model_names, metric_names, values = _collect_metric_values(reports)
    b = _resolve_backend(backend)
    return b.comparison_grouped_bar(model_names, metric_names, values)


def compare_and_plot_delta(
    models: dict[str, MatrixLike],
    *,
    baseline: str,
    labels: Optional[Sequence[int]] = None,
    hubness_k: int = 5,
    backend: Optional[PlotBackend] = None,
    **kwargs,
) -> object:
    reports = []
    for name, emb in models.items():
        r, _ = evaluate_embedding_quality(
            emb, labels=labels, hubness_k=hubness_k, model_name=name, **kwargs
        )
        reports.append(r)
    model_names, metric_names, values = _collect_metric_values(reports)

    base_idx = model_names.index(baseline) if baseline in model_names else 0
    base_vals = values[base_idx]

    delta_matrix: list[list[float]] = []
    for v in values:
        delta = [
            (v[i] - base_vals[i]) / abs(base_vals[i]) if base_vals[i] != 0 else 0.0
            for i in range(len(v))
        ]
        delta_matrix.append(delta)

    b = _resolve_backend(backend)
    return b.comparison_delta_heatmap(model_names, metric_names, delta_matrix)
