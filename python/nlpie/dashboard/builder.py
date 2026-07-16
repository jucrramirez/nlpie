from __future__ import annotations

from typing import Optional, Sequence

from nlpie._types import MatrixLike
from nlpie.metrics.quality import EmbeddingQualityReport, evaluate_embedding_quality
from ..backends import PlotBackend, PlotlyBackend


def _add_caption(fig, caption: str, y_pos: float = -0.18):
    if not caption:
        return
    fig.add_annotation(
        text=f"<span style='color:#666;font-size:10px'>{caption}</span>",
        xref="paper", yref="paper",
        x=0.5, y=y_pos,
        xanchor="center", yanchor="top",
        showarrow=False,
        align="center",
    )
    fig.update_layout(margin=dict(t=50, b=60, l=50, r=30))


def _reconstruct_similarity_matrix(pairwise: list[float], n: int) -> list[list[float]]:
    idx = 0
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            val = pairwise[idx]
            matrix[i][j] = val
            matrix[j][i] = val
            idx += 1
    return matrix


class FigureList(list):
    def show(self) -> None:
        for fig in self:
            fig.show()

    def _ipython_display_(self) -> None:
        for fig in self:
            fig._ipython_display_()


class DashboardBuilder:
    def __init__(self, report: Optional[EmbeddingQualityReport] = None, max_heatmap_size: int = 300):
        self._report = report
        self._backend: Optional[PlotBackend] = None
        self._sections: list[str] = []
        self._max_heatmap_size = max_heatmap_size

    @classmethod
    def from_embeddings(
        cls,
        embeddings: MatrixLike,
        labels: Optional[Sequence[int]] = None,
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

    def add_embedding_scatter(self) -> DashboardBuilder:
        self._sections.append("embedding_scatter")
        return self

    def build(self) -> FigureList:
        backend = self._backend if self._backend is not None else PlotlyBackend()

        if self._report is None:
            _, chart_list, _ = backend.full_dashboard(self._report)
            return FigureList(fig for _, fig in chart_list)

        report = self._report
        from nlpie.backends.plotly import _CHART_CAPTIONS

        chart_list: list[object] = []

        if "hubness_histogram" in self._sections and report.geometry is not None:
            from nlpie.dashboard import plot_hubness_histogram
            counts = report.hubness_counts or [0] * report.n_samples
            fig = plot_hubness_histogram(counts, report.geometry.hubness_skewness, report.geometry.hubness_k, backend=backend)
            _add_caption(fig, _CHART_CAPTIONS.get("Hubness Distribution", ""))
            chart_list.append(fig)

        if "similarity_distribution" in self._sections and report.intrinsic is not None:
            from nlpie.dashboard import plot_similarity_distribution
            values = report.pairwise_similarities or [0.0]
            fig = plot_similarity_distribution(values, report.intrinsic.mean, report.intrinsic.std, backend=backend)
            _add_caption(fig, _CHART_CAPTIONS.get("Pairwise Cosine Similarity", ""))
            chart_list.append(fig)

        if "similarity_heatmap" in self._sections:
            sim_matrix_2d = getattr(report, "similarity_matrix", None)
            if sim_matrix_2d is None and report.pairwise_similarities and report.n_samples > 0:
                sim_matrix_2d = _reconstruct_similarity_matrix(
                    report.pairwise_similarities, report.n_samples
                )
            if sim_matrix_2d is not None:
                n = len(sim_matrix_2d)
                if n > self._max_heatmap_size:
                    step = n // self._max_heatmap_size
                    indices = list(range(0, n, step))[: self._max_heatmap_size]
                    sim_matrix_2d = [
                        [sim_matrix_2d[i][j] for j in indices] for i in indices
                    ]
                from nlpie.dashboard import plot_similarity_heatmap
                fig = plot_similarity_heatmap(sim_matrix_2d, backend=backend)
                fig.update_layout(margin=dict(t=50, b=40, l=50, r=30))
                chart_list.append(fig)

        if "projection_quality" in self._sections and report.projection:
            from nlpie.dashboard import plot_projection_quality
            p = report.projection
            k_vals = [x.k for x in p]
            t_vals = [x.trustworthiness for x in p]
            c_vals = [x.continuity for x in p]
            fig = plot_projection_quality(k_vals, t_vals, c_vals, backend=backend)
            _add_caption(fig, _CHART_CAPTIONS.get("Projection Quality", ""), y_pos=-0.14)
            chart_list.append(fig)

        if "retrieval_metrics" in self._sections and report.retrieval:
            from nlpie.dashboard import plot_retrieval_metrics
            r = report.retrieval
            k_vals = [x.k for x in r]
            rec_vals = [x.recall for x in r]
            prec_vals = [x.precision for x in r]
            ndcg_vals = [x.ndcg for x in r]
            fig = plot_retrieval_metrics(k_vals, rec_vals, prec_vals, ndcg_vals, backend=backend)
            _add_caption(fig, _CHART_CAPTIONS.get("Retrieval Quality", ""), y_pos=-0.14)
            chart_list.append(fig)

        if "embedding_scatter" in self._sections:
            from nlpie.dashboard import plot_embedding_scatter
            if report.projection and len(report.projection) > 0:
                coords = getattr(report, "projection_coords", None)
                if coords is not None:
                    fig = plot_embedding_scatter([c[0] for c in coords], [c[1] for c in coords], backend=backend)
                    chart_list.append(fig)

        if not chart_list:
            _, chart_list, _ = backend.full_dashboard(report)
            return FigureList(fig for _, fig in chart_list)

        return FigureList(chart_list)
