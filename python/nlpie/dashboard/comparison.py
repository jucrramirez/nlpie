from __future__ import annotations

from collections.abc import Sequence

from nlpie._types import MatrixLike
from nlpie.metrics.quality import (
    EmbeddingQualityReport,
    evaluate_embedding_quality,
)

from ..backends import PlotBackend, _resolve_backend


def _evaluate_models(
    models: dict[str, MatrixLike],
    labels: Sequence[int] | None,
    hubness_k: int,
    **kwargs,
) -> list[EmbeddingQualityReport]:
    reports = []
    for name, emb in models.items():
        report, _ = evaluate_embedding_quality(
            emb, labels=labels, hubness_k=hubness_k, model_name=name, **kwargs
        )
        reports.append(report)
    return reports


def _collect_metric_values(
    reports: list[EmbeddingQualityReport],
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Align per-model metric values to a global metric column set.

    The column set is built from *all* reports first, then each model fills
    its row positionally — models missing a section get ``None`` in those
    columns, so values never silently shift out of alignment.
    """
    model_names = [r.model_name for r in reports]

    metric_names: list[str] = []

    def add(name: str) -> None:
        if name not in metric_names:
            metric_names.append(name)

    for r in reports:
        if r.intrinsic is not None:
            add("Intrinsic-Mean")
        if r.clustering is not None:
            add("ARI")
            add("NMI")
            add("Silhouette")
        if r.geometry is not None:
            add("Eff-Rank (frac)")
        if r.projection:
            add("Proj-Trust")
            add("Proj-Cont")
        if r.retrieval:
            add("R@K")
            add("nDCG")

    values: list[list[float | None]] = []
    for r in reports:
        col: dict[str, float | None] = {}
        if r.intrinsic is not None:
            col["Intrinsic-Mean"] = r.intrinsic.mean
        if r.clustering is not None:
            col["ARI"] = r.clustering.ari
            col["NMI"] = r.clustering.nmi
            col["Silhouette"] = r.clustering.silhouette
        if r.geometry is not None and r.n_dims > 0:
            # Normalise to a [0, 1] fraction of the ambient dimension so the
            # unbounded effective rank can share an axis with bounded metrics.
            col["Eff-Rank (frac)"] = min(r.geometry.effective_rank / r.n_dims, 1.0)
        if r.projection:
            col["Proj-Trust"] = sum(p.trustworthiness for p in r.projection) / len(r.projection)
            col["Proj-Cont"] = sum(p.continuity for p in r.projection) / len(r.projection)
        if r.retrieval:
            col["R@K"] = sum(rr.recall for rr in r.retrieval) / len(r.retrieval)
            col["nDCG"] = sum(rr.ndcg for rr in r.retrieval) / len(r.retrieval)
        values.append([col.get(name) for name in metric_names])

    return model_names, metric_names, values


def compare_and_plot_radar(
    models: dict[str, MatrixLike],
    *,
    labels: Sequence[int] | None = None,
    hubness_k: int = 5,
    backend: PlotBackend | None = None,
    **kwargs,
) -> object:
    reports = _evaluate_models(models, labels, hubness_k, **kwargs)
    model_names, metric_names, values = _collect_metric_values(reports)
    b = _resolve_backend(backend)
    return b.comparison_radar(model_names, metric_names, values)


def compare_and_plot_grouped_bar(
    models: dict[str, MatrixLike],
    *,
    labels: Sequence[int] | None = None,
    hubness_k: int = 5,
    backend: PlotBackend | None = None,
    **kwargs,
) -> object:
    reports = _evaluate_models(models, labels, hubness_k, **kwargs)
    model_names, metric_names, values = _collect_metric_values(reports)
    b = _resolve_backend(backend)
    return b.comparison_grouped_bar(model_names, metric_names, values)


def compare_and_plot_delta(
    models: dict[str, MatrixLike],
    *,
    baseline: str,
    labels: Sequence[int] | None = None,
    hubness_k: int = 5,
    backend: PlotBackend | None = None,
    **kwargs,
) -> object:
    reports = _evaluate_models(models, labels, hubness_k, **kwargs)
    model_names, metric_names, values = _collect_metric_values(reports)

    if baseline not in model_names:
        raise ValueError(f"Unknown baseline {baseline!r}; available models: {model_names}")
    base_vals = values[model_names.index(baseline)]

    delta_matrix: list[list[float | None]] = []
    for v in values:
        delta: list[float | None] = []
        for val, base in zip(v, base_vals):
            if val is None or base is None or base == 0.0:
                delta.append(None)
            else:
                delta.append((val - base) / abs(base))
        delta_matrix.append(delta)

    b = _resolve_backend(backend)
    return b.comparison_delta_heatmap(model_names, metric_names, delta_matrix)
