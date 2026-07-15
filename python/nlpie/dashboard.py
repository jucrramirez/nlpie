from __future__ import annotations

from typing import Optional, Sequence

from .backends import PlotBackend, PlotlyBackend


def plot_hubness_histogram(
    counts: Sequence[int],
    skewness: float,
    k: int,
    backend: Optional[PlotBackend] = None,
) -> object:
    if backend is None:
        backend = PlotlyBackend()
    return backend.hubness_histogram(counts, skewness, k)


def plot_similarity_distribution(
    values: Sequence[float],
    mean: float,
    std: float,
    backend: Optional[PlotBackend] = None,
) -> object:
    if backend is None:
        backend = PlotlyBackend()
    return backend.similarity_distribution(values, mean, std)


def plot_similarity_to_mean(
    sims: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    if backend is None:
        backend = PlotlyBackend()
    return backend.similarity_to_mean_chart(sims)


def plot_projection_quality(
    k_values: Sequence[int],
    trustworthiness: Sequence[float],
    continuity: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    if backend is None:
        backend = PlotlyBackend()
    return backend.projection_quality_curve(k_values, trustworthiness, continuity)


def plot_retrieval_metrics(
    k_values: Sequence[int],
    recall: Sequence[float],
    precision: Sequence[float],
    ndcg: Sequence[float],
    backend: Optional[PlotBackend] = None,
) -> object:
    if backend is None:
        backend = PlotlyBackend()
    return backend.retrieval_metrics_curve(k_values, recall, precision, ndcg)
