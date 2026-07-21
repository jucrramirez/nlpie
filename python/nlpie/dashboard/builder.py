from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from nlpie._similarity import downsample_square_matrix, reconstruct_similarity_matrix
from nlpie._types import MatrixLike
from nlpie.metrics.quality import EmbeddingQualityReport, evaluate_embedding_quality

from .._dashboard import (
    plot_hubness_histogram,
    plot_projection_quality,
    plot_retrieval_metrics,
    plot_similarity_distribution,
    plot_similarity_heatmap,
)
from ..backends import PlotBackend, _resolve_backend
from ..backends.plotly import _CHART_CAPTIONS, _add_caption


class FigureList(list):
    def show(self) -> None:
        for fig in self:
            fig.show()

    def _ipython_display_(self) -> None:
        for fig in self:
            fig._ipython_display_()


class DashboardBuilder:
    def __init__(self, report: EmbeddingQualityReport | None = None, max_heatmap_size: int = 300):
        self._report = report
        self._backend: PlotBackend | None = None
        self._sections: list[str] = []
        self._max_heatmap_size = max_heatmap_size

    @classmethod
    def from_embeddings(
        cls,
        embeddings: MatrixLike,
        labels: Sequence[int] | None = None,
        max_heatmap_size: int = 300,
        **kwargs,
    ) -> DashboardBuilder:
        builder = cls(max_heatmap_size=max_heatmap_size)
        report, _ = evaluate_embedding_quality(embeddings, labels=labels, **kwargs)
        builder._report = report
        return builder

    def with_backend(self, backend: PlotBackend) -> DashboardBuilder:
        self._backend = backend
        return self

    def add_hubness_histogram(self) -> DashboardBuilder:
        self._sections.append("hubness_histogram")
        return self

    def add_similarity_distribution(self) -> DashboardBuilder:
        self._sections.append("similarity_distribution")
        return self

    def add_similarity_heatmap(self) -> DashboardBuilder:
        self._sections.append("similarity_heatmap")
        return self

    def add_projection_quality(self) -> DashboardBuilder:
        self._sections.append("projection_quality")
        return self

    def add_retrieval_metrics(self) -> DashboardBuilder:
        self._sections.append("retrieval_metrics")
        return self

    def build(self) -> FigureList:
        if self._report is None:
            raise ValueError(
                "DashboardBuilder has no report. Pass one to the constructor or "
                "use DashboardBuilder.from_embeddings(...)."
            )

        backend = _resolve_backend(self._backend)
        report = self._report
        chart_list: list[object] = []

        if "hubness_histogram" in self._sections and report.geometry is not None:
            counts = report.hubness_counts or [0] * report.n_samples
            fig = plot_hubness_histogram(
                counts, report.geometry.hubness_skewness, report.geometry.hubness_k, backend=backend
            )
            _add_caption(fig, _CHART_CAPTIONS.get("Hubness Distribution", ""))
            chart_list.append(fig)

        if "similarity_distribution" in self._sections and report.intrinsic is not None:
            values = report.pairwise_similarities or [0.0]
            fig = plot_similarity_distribution(
                values, report.intrinsic.mean, report.intrinsic.std, backend=backend
            )
            _add_caption(fig, _CHART_CAPTIONS.get("Pairwise Cosine Similarity", ""))
            chart_list.append(fig)

        if (
            "similarity_heatmap" in self._sections
            and report.pairwise_similarities
            and report.n_samples > 1
        ):
            sim_matrix_2d = reconstruct_similarity_matrix(
                report.pairwise_similarities, report.n_samples
            )
            sim_matrix_2d = downsample_square_matrix(sim_matrix_2d, self._max_heatmap_size)
            fig = cast(Any, plot_similarity_heatmap(sim_matrix_2d, backend=backend))
            fig.update_layout(margin=dict(t=50, b=40, l=50, r=30))
            chart_list.append(fig)

        if "projection_quality" in self._sections and report.projection:
            p = report.projection
            fig = plot_projection_quality(
                [x.k for x in p],
                [x.trustworthiness for x in p],
                [x.continuity for x in p],
                backend=backend,
            )
            _add_caption(fig, _CHART_CAPTIONS.get("Projection Quality", ""), y_pos=-0.14)
            chart_list.append(fig)

        if "retrieval_metrics" in self._sections and report.retrieval:
            r = report.retrieval
            fig = plot_retrieval_metrics(
                [x.k for x in r],
                [x.recall for x in r],
                [x.precision for x in r],
                [x.ndcg for x in r],
                backend=backend,
            )
            _add_caption(fig, _CHART_CAPTIONS.get("Retrieval Quality", ""), y_pos=-0.14)
            chart_list.append(fig)

        if not chart_list:
            dashboard = backend.full_dashboard(report)
            return FigureList(fig for _, fig in dashboard.charts)

        return FigureList(chart_list)
