from __future__ import annotations

from typing import Optional, Sequence

from .backends import PlotBackend, PlotlyBackend


def _resolve_backend(backend: Optional[PlotBackend] = None) -> PlotBackend:
    if backend is None:
        return PlotlyBackend()
    return backend


def plot_hubness_histogram(
    counts: Sequence[int],
    skewness: float,
    k: int,
    backend: Optional[PlotBackend] = None,
) -> object:
    return _resolve_backend(backend).hubness_histogram(counts, skewness, k)


def plot_similarity_distribution(
    values: Sequence[float],
    mean: float,
    std: float,
    backend: Optional[PlotBackend] = None,
) -> object:
    return _resolve_backend(backend).similarity_distribution(values, mean, std)


def plot_similarity_to_mean(
    sims: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    return _resolve_backend(backend).similarity_to_mean_chart(sims)


def plot_projection_quality(
    k_values: Sequence[int],
    trustworthiness: Sequence[float],
    continuity: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    return _resolve_backend(backend).projection_quality_curve(k_values, trustworthiness, continuity)


def plot_retrieval_metrics(
    k_values: Sequence[int],
    recall: Sequence[float],
    precision: Sequence[float],
    ndcg: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    return _resolve_backend(backend).retrieval_metrics_curve(k_values, recall, precision, ndcg)


def plot_similarity_heatmap(
    sim_matrix: list[list[float]],
    labels: Optional[Sequence[int]] = None,
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.similarity_heatmap(sim_matrix, labels)


def plot_hubness_bar(
    counts: Sequence[int],
    top_n: int = 50,
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.hubness_bar_chart(counts, top_n)


def plot_eigenvalue_scree(
    eigenvalues: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.eigenvalue_scree_plot(eigenvalues)


def plot_silhouette(
    silhouette_values: Sequence[float],
    labels: Sequence[int],
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.silhouette_plot(silhouette_values, labels)


def plot_correlation_heatmap(
    corr_matrix: list[list[float]],
    labels: list[str],
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.correlation_heatmap(corr_matrix, labels)


def plot_embedding_scatter(
    x: Sequence[float],
    y: Sequence[float],
    labels: Optional[Sequence[int]] = None,
    title: str = "Embedding Space (2D projection)",
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.embedding_scatter(x, y, labels, title)


def plot_comparison_radar(
    model_names: list[str],
    metric_names: list[str],
    values: list[list[float]],
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.comparison_radar(model_names, metric_names, values)


def plot_comparison_grouped_bar(
    model_names: list[str],
    metric_names: list[str],
    values: list[list[float]],
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.comparison_grouped_bar(model_names, metric_names, values)


def plot_comparison_delta(
    model_names: list[str],
    metric_names: list[str],
    delta_matrix: list[list[float]],
    backend: Optional[PlotBackend] = None,
) -> object:
    b = _resolve_backend(backend)
    return b.comparison_delta_heatmap(model_names, metric_names, delta_matrix)


def plot_quality_report(
    report,
    backend: Optional[PlotBackend] = None,
    interpretation=None,
) -> tuple:
    return _resolve_backend(backend).full_dashboard(report, interpretation=interpretation)
