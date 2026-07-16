from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class PlotBackend(ABC):
    @abstractmethod
    def hubness_histogram(self, counts: Sequence[int], skewness: float, k: int) -> object:
        ...

    @abstractmethod
    def similarity_distribution(self, values: Sequence[float], mean: float, std: float) -> object:
        ...

    @abstractmethod
    def similarity_to_mean_chart(self, sims: Sequence[float]) -> object:
        ...

    @abstractmethod
    def projection_quality_curve(
        self, k_values: Sequence[int], trustworthiness: Sequence[float], continuity: Sequence[float]
    ) -> object:
        ...

    @abstractmethod
    def retrieval_metrics_curve(
        self,
        k_values: Sequence[int],
        recall: Sequence[float],
        precision: Sequence[float],
        ndcg: Sequence[float],
    ) -> object:
        ...

    @abstractmethod
    def full_dashboard(self, report, interpretation=None) -> tuple:
        ...
