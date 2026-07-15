from __future__ import annotations

from typing import Sequence

from .base import PlotBackend


class PlotlyBackend(PlotBackend):
    def __init__(self) -> None:
        try:
            import plotly.graph_objects as go
            self._go = go
            import plotly.subplots
            self._subplots = plotly.subplots
        except ImportError as e:
            raise ImportError(
                "Plotly is required for the PlotlyBackend. "
                "Install it with: uv pip install 'nlpie[plotting]'"
            ) from e

    def hubness_histogram(self, counts: Sequence[int], skewness: float, k: int) -> object:
        fig = self._go.Figure()
        fig.add_trace(self._go.Histogram(
            x=list(counts),
            nbinsx=max(counts) - min(counts) + 1 if counts else 1,
            name="Hubness counts",
        ))
        fig.add_vline(
            x=k,
            line_dash="dash",
            line_color="red",
            annotation_text=f"mean (k={k})",
        )
        fig.update_layout(
            title=f"Hubness Distribution (skewness={skewness:.3f})",
            xaxis_title="K-NN occurrence count",
            yaxis_title="Frequency",
            bargap=0.1,
        )
        return fig

    def similarity_distribution(self, values: Sequence[float], mean: float, std: float) -> object:
        fig = self._go.Figure()
        fig.add_trace(self._go.Histogram(
            x=list(values),
            name="Pairwise cosine similarity",
        ))
        fig.add_vline(
            x=mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"mean={mean:.4f}",
        )
        fig.update_layout(
            title=f"Cosine Similarity Distribution (std={std:.4f})",
            xaxis_title="Cosine similarity",
            yaxis_title="Frequency",
            bargap=0.1,
        )
        return fig

    def similarity_to_mean_chart(self, sims: Sequence[float]) -> object:
        sorted_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        sorted_sims = [sims[i] for i in sorted_indices]

        fig = self._go.Figure()
        fig.add_trace(self._go.Bar(
            x=list(range(len(sorted_sims))),
            y=sorted_sims,
            name="Similarity to global mean",
        ))
        fig.update_layout(
            title="Similarity of Each Point to the Global Centroid",
            xaxis_title="Point index (sorted by similarity)",
            yaxis_title="Cosine similarity",
        )
        return fig

    def projection_quality_curve(
        self,
        k_values: Sequence[int],
        trustworthiness: Sequence[float],
        continuity: Sequence[float],
    ) -> object:
        fig = self._go.Figure()
        fig.add_trace(self._go.Scatter(
            x=list(k_values),
            y=list(trustworthiness),
            mode="lines+markers",
            name="Trustworthiness",
        ))
        fig.add_trace(self._go.Scatter(
            x=list(k_values),
            y=list(continuity),
            mode="lines+markers",
            name="Continuity",
        ))
        fig.update_layout(
            title="Projection Quality vs. Neighbourhood Size",
            xaxis_title="K",
            yaxis_title="Score",
            yaxis_range=[0, 1],
        )
        return fig

    def retrieval_metrics_curve(
        self,
        k_values: Sequence[int],
        recall: Sequence[float],
        precision: Sequence[float],
        ndcg: Sequence[float],
    ) -> object:
        fig = self._go.Figure()
        fig.add_trace(self._go.Scatter(
            x=list(k_values),
            y=list(recall),
            mode="lines+markers",
            name="Recall@K",
        ))
        fig.add_trace(self._go.Scatter(
            x=list(k_values),
            y=list(precision),
            mode="lines+markers",
            name="Precision@K",
        ))
        fig.add_trace(self._go.Scatter(
            x=list(k_values),
            y=list(ndcg),
            mode="lines+markers",
            name="nDCG@K",
        ))
        fig.update_layout(
            title="Retrieval Metrics vs. Cut-off K",
            xaxis_title="K",
            yaxis_title="Score",
            yaxis_range=[0, 1],
        )
        return fig
