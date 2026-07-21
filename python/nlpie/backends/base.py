from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dashboard:
    """Structured result of a full quality dashboard render.

    Attributes:
        kpi: Figure holding the key-performance-indicator cards.
        charts: ``(title, figure)`` pairs for every metric chart.
        story: Storytelling figure with the analysis & recommendations,
            or ``None`` when no interpretation was provided.
    """

    kpi: Any = None
    charts: list[tuple[str, Any]] = field(default_factory=list)
    story: Any | None = None

    def figures(self) -> list[Any]:
        """Return all non-None figures in display order (kpi, charts, story)."""
        figs = []
        if self.kpi is not None:
            figs.append(self.kpi)
        figs.extend(fig for _, fig in self.charts)
        if self.story is not None:
            figs.append(self.story)
        return figs

    def show(self) -> None:
        for fig in self.figures():
            fig.show()

    def _ipython_display_(self) -> None:
        for fig in self.figures():
            fig._ipython_display_()


class PlotBackend(ABC):
    """Contract every plotting backend must implement.

    The public ``plot_*`` helpers and the dashboard/export machinery call
    exactly these methods, so an alternate backend (e.g. matplotlib) only
    needs to implement this interface.
    """

    # --- Single-metric charts ------------------------------------------------

    @abstractmethod
    def hubness_histogram(self, counts: Sequence[int], skewness: float, k: int) -> object: ...

    @abstractmethod
    def similarity_distribution(
        self, values: Sequence[float], mean: float, std: float
    ) -> object: ...

    @abstractmethod
    def similarity_to_mean_chart(self, sims: Sequence[float]) -> object: ...

    @abstractmethod
    def similarity_heatmap(
        self, sim_matrix: list[list[float]], labels: Sequence[int] | None = None
    ) -> object: ...

    @abstractmethod
    def hubness_bar_chart(self, counts: Sequence[int], top_n: int = 50) -> object: ...

    @abstractmethod
    def eigenvalue_scree_plot(self, eigenvalues: Sequence[float]) -> object: ...

    @abstractmethod
    def silhouette_plot(
        self, silhouette_values: Sequence[float], labels: Sequence[int]
    ) -> object: ...

    @abstractmethod
    def correlation_heatmap(self, corr_matrix: list[list[float]], labels: list[str]) -> object: ...

    @abstractmethod
    def embedding_scatter(
        self,
        x: Sequence[float],
        y: Sequence[float],
        labels: Sequence[int] | None = None,
        title: str = "Embedding Space (2D projection)",
    ) -> object: ...

    # --- Multi-metric curves ---------------------------------------------------

    @abstractmethod
    def projection_quality_curve(
        self, k_values: Sequence[int], trustworthiness: Sequence[float], continuity: Sequence[float]
    ) -> object: ...

    @abstractmethod
    def retrieval_metrics_curve(
        self,
        k_values: Sequence[int],
        recall: Sequence[float],
        precision: Sequence[float],
        ndcg: Sequence[float],
    ) -> object: ...

    # --- Model comparison ------------------------------------------------------

    @abstractmethod
    def comparison_radar(
        self,
        model_names: list[str],
        metric_names: list[str],
        values: list[list[float | None]],
    ) -> object: ...

    @abstractmethod
    def comparison_grouped_bar(
        self,
        model_names: list[str],
        metric_names: list[str],
        values: list[list[float | None]],
    ) -> object: ...

    @abstractmethod
    def comparison_delta_heatmap(
        self,
        model_names: list[str],
        metric_names: list[str],
        delta_matrix: list[list[float | None]],
    ) -> object: ...

    # --- Full dashboard ----------------------------------------------------------

    @abstractmethod
    def full_dashboard(self, report, interpretation=None) -> Dashboard: ...
