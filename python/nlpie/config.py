from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NlpieConfig:
    hubness_severe_threshold: float = 1.0
    hubness_moderate_threshold: float = 0.5
    default_hubness_k: int = 5
    default_projection_k_values: tuple[int, ...] = (5, 10, 20)
    default_retrieval_k_values: tuple[int, ...] = (1, 5, 10)
    plotly_theme: str = "plotly_white"
