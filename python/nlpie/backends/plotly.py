from __future__ import annotations

from typing import Optional, Sequence

from .base import PlotBackend


_SEVERITY_COLORS = {
    "critical": "#d62728",
    "warning": "#ff7f0e",
    "info": "#2ca02c",
}


def _axis_layout_key(subplot_num: int, axis: str) -> str:
    return axis if subplot_num == 1 else f"{axis}{subplot_num}"


def _subplot_number(row: int, col: int, n_cols: int) -> int:
    return (row - 1) * n_cols + col


def _compute_subplot_num(specs, row: int, col: int) -> int:
    num = 0
    for r, spec_row in enumerate(specs, 1):
        c = 1
        for spec in spec_row:
            if spec is None:
                c += 1
                continue
            num += 1
            if r == row and c == col:
                return num
            c += spec.get("colspan", 1)
    raise ValueError(f"No subplot at row={row}, col={col}")


def _get_subplot_domain(fig, subplot_num: int) -> tuple[tuple[float, float], tuple[float, float]]:
    x_key = _axis_layout_key(subplot_num, "xaxis")
    y_key = _axis_layout_key(subplot_num, "yaxis")
    xdom = tuple(fig.layout[x_key].domain)
    ydom = tuple(fig.layout[y_key].domain)
    return xdom, ydom


def _paper_position(
    xdom: tuple[float, float],
    ydom: tuple[float, float],
    x_frac: float,
    y_frac: float,
) -> tuple[float, float]:
    return (
        xdom[0] + x_frac * (xdom[1] - xdom[0]),
        ydom[0] + y_frac * (ydom[1] - ydom[0]),
    )



def _kpi_severity(metric: str, value: float, n_dims: int = 0) -> str:
    if metric == "mean":
        if value > 0.8: return _SEVERITY_COLORS["critical"]
        if value > 0.5: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric == "std":
        if value < 0.05: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric == "eff_rank":
        if n_dims > 0 and value < n_dims * 0.3: return _SEVERITY_COLORS["critical"]
        if n_dims > 0 and value < n_dims * 0.6: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric == "hubness":
        if value > 1.0: return _SEVERITY_COLORS["critical"]
        if value > 0.5: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric == "ari":
        if value < 0.25: return _SEVERITY_COLORS["critical"]
        if value < 0.5: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric == "silhouette":
        if value < 0.1: return _SEVERITY_COLORS["critical"]
        if value < 0.25: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric in ("trust", "cont"):
        if value < 0.5: return _SEVERITY_COLORS["critical"]
        if value < 0.8: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    if metric in ("recall", "ndcg"):
        if value < 0.3: return _SEVERITY_COLORS["critical"]
        if value < 0.7: return _SEVERITY_COLORS["warning"]
        return _SEVERITY_COLORS["info"]
    return "#999"


def _kpi_cards_data(report) -> list[tuple[str, str, str]]:
    n_dims = getattr(report, "n_dims", 0)
    cards: list[tuple[str, str, str]] = []

    m = getattr(report, "intrinsic", None)
    if m is not None:
        cards.append(("Mean", f"{m.mean:.3f}", _kpi_severity("mean", m.mean)))
        cards.append(("Std", f"{m.std:.3f}", _kpi_severity("std", m.std)))

    m = getattr(report, "geometry", None)
    if m is not None:
        cards.append(("Eff-Rank", f"{m.effective_rank:.1f}", _kpi_severity("eff_rank", m.effective_rank, n_dims)))
        cards.append(("Hubness", f"{m.hubness_skewness:.2f}", _kpi_severity("hubness", m.hubness_skewness)))

    m = getattr(report, "clustering", None)
    if m is not None:
        cards.append(("ARI", f"{m.ari:.3f}", _kpi_severity("ari", m.ari)))
        cards.append(("Silhouette", f"{m.silhouette:.3f}", _kpi_severity("silhouette", m.silhouette)))

    if getattr(report, "projection", None):
        p = report.projection
        avg_t = sum(x.trustworthiness for x in p) / len(p)
        avg_c = sum(x.continuity for x in p) / len(p)
        cards.append(("Trust", f"{avg_t:.3f}", _kpi_severity("trust", avg_t)))
        cards.append(("Cont", f"{avg_c:.3f}", _kpi_severity("cont", avg_c)))

    if getattr(report, "retrieval", None):
        r = report.retrieval
        avg_rec = sum(x.recall for x in r) / len(r)
        avg_ndcg = sum(x.ndcg for x in r) / len(r)
        cards.append(("Recall", f"{avg_rec:.3f}", _kpi_severity("recall", avg_rec)))
        cards.append(("NDCG", f"{avg_ndcg:.3f}", _kpi_severity("ndcg", avg_ndcg)))

    return cards


def _build_kpi_figure(go, report) -> object:
    cards = _kpi_cards_data(report)
    fig = go.Figure()

    if cards:
        n = len(cards)
        cw = 0.88 / n
        margin_x = 0.06
        label_size = 11 if n <= 6 else 9
        value_size = 22 if n <= 6 else 16

        for i, (label, value, color) in enumerate(cards):
            cx = margin_x + cw * (i + 0.5)
            left = cx - cw * 0.42

            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=left, y0=0.15, x1=left + 0.005, y1=0.78,
                fillcolor=color,
                line=dict(width=0),
            )
            fig.add_annotation(
                text=label,
                xref="paper", yref="paper",
                x=cx, y=0.70,
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=label_size, color="#888"),
            )
            fig.add_annotation(
                text=value,
                xref="paper", yref="paper",
                x=cx, y=0.40,
                xanchor="center", yanchor="middle",
                showarrow=False,
                font=dict(size=value_size, color="#1a1a2e", family="Arial Black, sans-serif"),
            )

    model_name = getattr(report, "model_name", "unnamed")
    n_samples = getattr(report, "n_samples", 0)
    n_dims = getattr(report, "n_dims", 0)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Key Performance Indicators</b><br>"
                f"<span style='font-size:11px;color:#555'>{model_name} — {n_samples:,} samples · {n_dims:,} dimensions</span>"
            ),
            x=0.5, xanchor="center",
        ),
        height=170,
        margin=dict(t=48, b=5, l=10, r=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _format_storytelling(explanations) -> str:
    if not explanations:
        return '<span style="color:#2ca02c;font-size:12px;font-weight:bold">\u2713 All metrics look healthy \u2014 no issues detected.</span>'

    n_crit = sum(1 for e in explanations if e.severity == "critical")
    n_warn = sum(1 for e in explanations if e.severity == "warning")
    n_info = sum(1 for e in explanations if e.severity == "info")

    counts = ", ".join(
        s for s in [
            f"{n_crit} critical" if n_crit else "",
            f"{n_warn} warning" if n_warn else "",
            f"{n_info} info" if n_info else "",
        ] if s
    )

    parts = [
        '<span style="font-size:13px;font-weight:bold;color:#333">Assessment Summary</span><br>'
        f'<span style="font-size:11px;color:#555">Detected {counts} issue(s) in the embedding space.</span>'
    ]

    for severity, header in [
        ("critical", "Critical Issues"),
        ("warning", "Warnings"),
        ("info", "Info"),
    ]:
        items = [e for e in explanations if e.severity == severity]
        if not items:
            continue
        color = _SEVERITY_COLORS[severity]
        parts.append(
            f'<br><br><span style="font-size:12px;font-weight:bold;color:{color}">\u26a0 {header}</span>'
        )
        for exp in items:
            parts.append(
                f'<br><span style="font-size:11px;color:#333"><b>{exp.metric}</b>: {exp.summary}</span>'
            )
            if exp.recommendation:
                parts.append(
                    '<br><span style="font-size:10px;color:#555;padding-left:14px;display:inline-block">'
                    f'\u2192 {exp.recommendation}</span>'
                )

    recs = [e.recommendation for e in explanations if e.recommendation]
    if recs:
        parts.append(
            '<br><br><span style="font-size:12px;font-weight:bold;color:#333">Recommended Actions</span><br>'
        )
        for i, rec in enumerate(recs, 1):
            parts.append(
                f'<span style="font-size:10px;color:#444">{i}. {rec}</span><br>'
            )

    return "".join(parts)


def _copy_traces(fig, source_fig, row: int, col: int) -> None:
    for trace in source_fig.data:
        fig.add_trace(trace, row=row, col=col)


def _hide_axes(fig, row: int, col: int) -> None:
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False, row=row, col=col)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False, row=row, col=col)


def _add_domain_text(
    fig,
    subplot_num: int,
    text: str,
    *,
    x_frac: float = 0.03,
    y_frac: float = 0.78,
    font_size: int = 11,
) -> None:
    xdom, ydom = _get_subplot_domain(fig, subplot_num)
    x, y = _paper_position(xdom, ydom, x_frac, y_frac)
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=font_size),
        align="left",
    )


def _add_domain_caption(
    fig,
    subplot_num: int,
    text: str,
) -> None:
    xdom, ydom = _get_subplot_domain(fig, subplot_num)
    x, y = _paper_position(xdom, ydom, 0.04, 0.10)
    fig.add_annotation(
        text=f"<span style='color:#666;font-size:10px'>{text}</span>",
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        align="left",
    )


def _estimate_dashboard_height(
    n_charts: int,
    include_interp: bool,
    explanation_count: int,
) -> int:
    kpi_px = 80
    chart_px = 280 * max(1, (n_charts + 1) // 2) if n_charts > 0 else 0
    interp_px = (120 + explanation_count * 60) if include_interp else 0
    total_px = 120 + kpi_px + chart_px + interp_px
    return max(total_px, 600)


_CHART_CAPTIONS = {
    "Pairwise Cosine Similarity": "Peak near 1.0 = redundant vectors; wide spread = diverse space",
    "Hubness Distribution": "Right-skewed = a few points dominate neighbour lists",
    "Projection Quality": "Scores near 1.0 mean the projection preserves local structure",
    "Retrieval Quality": "Higher curves = better nearest-neighbour retrieval",
}


class PlotlyBackend(PlotBackend):
    def __init__(self) -> None:
        try:
            import plotly.graph_objects as go
            self._go = go
            import plotly.subplots
            self._subplots = plotly.subplots
            import plotly.express as px
            self._px = px
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
            margin=dict(t=60, b=50),
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
            margin=dict(t=60, b=50),
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

    def similarity_heatmap(
        self,
        sim_matrix: list[list[float]],
        labels: Optional[Sequence[int]] = None,
    ) -> object:
        n = len(sim_matrix)
        if labels is not None:
            order = sorted(range(n), key=lambda i: labels[i])
            sorted_matrix = [[sim_matrix[i][j] for j in order] for i in order]
            sorted_labels = [labels[i] for i in order]
            x_labels = [str(l) for l in sorted_labels]
            y_labels = [str(l) for l in sorted_labels]
        else:
            sorted_matrix = sim_matrix
            x_labels = list(range(n))
            y_labels = list(range(n))

        fig = self._go.Figure(data=self._go.Heatmap(
            z=sorted_matrix,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1, zmax=1,
        ))
        fig.update_layout(
            title="Pairwise Cosine Similarity",
            xaxis_title="Point index" if labels is None else "Label",
            yaxis_title="Point index" if labels is None else "Label",
            width=600,
            height=600,
        )
        return fig

    def hubness_bar_chart(self, counts: Sequence[int], top_n: int = 50) -> object:
        sorted_indices = sorted(
            range(len(counts)), key=lambda i: counts[i], reverse=True
        )[:top_n]
        sorted_counts = [counts[i] for i in sorted_indices]

        fig = self._go.Figure()
        fig.add_trace(self._go.Bar(
            x=[str(i) for i in sorted_indices],
            y=sorted_counts,
            name="Hubness count",
        ))
        fig.update_layout(
            title=f"Hubness per Point (top {min(top_n, len(counts))})",
            xaxis_title="Point index",
            yaxis_title="K-NN occurrence count",
        )
        return fig

    def eigenvalue_scree_plot(self, eigenvalues: Sequence[float]) -> object:
        ev = list(eigenvalues)
        cumulative = [sum(ev[:i+1]) / sum(ev) for i in range(len(ev))] if sum(ev) > 0 else []

        fig = self._go.Figure()
        fig.add_trace(self._go.Bar(
            x=list(range(1, len(ev) + 1)),
            y=ev,
            name="Eigenvalue",
        ))
        if cumulative:
            fig.add_trace(self._go.Scatter(
                x=list(range(1, len(cumulative) + 1)),
                y=cumulative,
                mode="lines+markers",
                name="Cumulative variance",
                yaxis="y2",
            ))
        fig.update_layout(
            title="Eigenvalue Scree Plot",
            xaxis_title="Component",
            yaxis_title="Eigenvalue",
            yaxis2=dict(
                title="Cumulative variance ratio",
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
        )
        return fig

    def silhouette_plot(
        self,
        silhouette_values: Sequence[float],
        labels: Sequence[int],
    ) -> object:
        n = len(silhouette_values)
        mean_sil = sum(silhouette_values) / n if n > 0 else 0.0

        order = sorted(range(n), key=lambda i: (labels[i], silhouette_values[i]))
        ordered_vals = [silhouette_values[i] for i in order]
        ordered_labels = [labels[i] for i in order]

        fig = self._go.Figure()
        fig.add_trace(self._go.Bar(
            x=list(range(n)),
            y=ordered_vals,
            marker=dict(color=ordered_labels, colorscale="Viridis"),
            showlegend=False,
        ))
        fig.add_hline(
            y=mean_sil,
            line_dash="dash",
            line_color="red",
            annotation_text=f"mean={mean_sil:.3f}",
        )
        fig.update_layout(
            title="Silhouette Plot",
            xaxis_title="Point (sorted by cluster + silhouette)",
            yaxis_title="Silhouette coefficient",
            yaxis_range=[-1, 1],
        )
        return fig

    def correlation_heatmap(self, corr_matrix: list[list[float]], labels: list[str]) -> object:
        fig = self._go.Figure(data=self._go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Metric Correlation Matrix",
            width=600,
            height=600,
        )
        return fig

    def embedding_scatter(
        self,
        x: Sequence[float],
        y: Sequence[float],
        labels: Optional[Sequence[int]] = None,
        title: str = "Embedding Space (2D projection)",
    ) -> object:
        fig = self._go.Figure()
        fig.add_trace(self._go.Scatter(
            x=list(x),
            y=list(y),
            mode="markers",
            marker=dict(
                color=list(labels) if labels is not None else None,
                colorscale="Viridis",
                showscale=labels is not None,
                size=6,
            ),
            text=[str(l) for l in labels] if labels is not None else None,
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode="closest",
        )
        return fig

    def embedding_scatter_3d(
        self,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float],
        labels: Optional[Sequence[int]] = None,
        title: str = "Embedding Space (3D projection)",
    ) -> object:
        fig = self._go.Figure(data=[self._go.Scatter3d(
            x=list(x),
            y=list(y),
            z=list(z),
            mode="markers",
            marker=dict(
                color=list(labels) if labels is not None else None,
                colorscale="Viridis",
                showscale=labels is not None,
                size=4,
            ),
        )])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
            ),
        )
        return fig

    def comparison_radar(
        self,
        model_names: list[str],
        metric_names: list[str],
        values: list[list[float]],
    ) -> object:
        fig = self._go.Figure()
        for name, vals in zip(model_names, values):
            fig.add_trace(self._go.Scatterpolar(
                r=vals + [vals[0]],
                theta=metric_names + [metric_names[0]],
                fill="toself",
                name=name,
            ))
        fig.update_layout(
            title="Model Comparison — Radar Chart",
            polar=dict(radialaxis=dict(range=[0, 1])),
        )
        return fig

    def comparison_grouped_bar(
        self,
        model_names: list[str],
        metric_names: list[str],
        values: list[list[float]],
    ) -> object:
        fig = self._go.Figure()
        for idx, name in enumerate(model_names):
            fig.add_trace(self._go.Bar(
                name=name,
                x=metric_names,
                y=values[idx],
            ))
        fig.update_layout(
            title="Model Comparison — Grouped Bar Chart",
            barmode="group",
            yaxis_title="Score",
            yaxis_range=[0, 1],
        )
        return fig

    def comparison_delta_heatmap(
        self,
        model_names: list[str],
        metric_names: list[str],
        delta_matrix: list[list[float]],
    ) -> object:
        fig = self._go.Figure(data=self._go.Heatmap(
            z=delta_matrix,
            x=metric_names,
            y=model_names,
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:+.1%}" for v in row] for row in delta_matrix],
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Model Comparison — Delta (%)",
            xaxis_title="Metric",
            yaxis_title="Model",
        )
        return fig

    def _build_chart_list(self, report) -> list[tuple[str, object]]:
        charts: list[tuple[str, object]] = []

        if getattr(report, "intrinsic", None) is not None:
            sims = getattr(report, "pairwise_similarities", None) or []
            if sims:
                fig = self.similarity_distribution(sims, report.intrinsic.mean, report.intrinsic.std)
                cap = _CHART_CAPTIONS.get("Pairwise Cosine Similarity", "")
                fig.add_annotation(
                    text=f"<span style='color:#666;font-size:10px'>{cap}</span>",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.18,
                    xanchor="center", yanchor="top",
                    showarrow=False,
                    align="center",
                )
                fig.update_layout(margin=dict(t=50, b=60, l=50, r=30))
                charts.append(("Pairwise Cosine Similarity", fig))

        if getattr(report, "geometry", None) is not None:
            counts = getattr(report, "hubness_counts", None) or []
            if counts:
                fig = self.hubness_histogram(counts, report.geometry.hubness_skewness, report.geometry.hubness_k)
                cap = _CHART_CAPTIONS.get("Hubness Distribution", "")
                fig.add_annotation(
                    text=f"<span style='color:#666;font-size:10px'>{cap}</span>",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.18,
                    xanchor="center", yanchor="top",
                    showarrow=False,
                    align="center",
                )
                fig.update_layout(margin=dict(t=50, b=60, l=50, r=30))
                charts.append(("Hubness Distribution", fig))

        pairwise = getattr(report, "pairwise_similarities", None) or []
        n = getattr(report, "n_samples", 0)
        if pairwise and n > 0 and len(pairwise) >= n - 1:
            from nlpie.dashboard.builder import _reconstruct_similarity_matrix
            matrix = _reconstruct_similarity_matrix(pairwise, n)
            max_n = 300
            if n > max_n:
                step = n // max_n
                indices = list(range(0, n, step))[:max_n]
                matrix = [[matrix[i][j] for j in indices] for i in indices]
            fig = self.similarity_heatmap(matrix)
            fig.update_layout(margin=dict(t=50, b=40, l=50, r=30))
            charts.append(("Similarity Heatmap", fig))

        if getattr(report, "projection", None):
            p = report.projection
            if p:
                fig = self.projection_quality_curve(
                    [x.k for x in p], [x.trustworthiness for x in p], [x.continuity for x in p],
                )
                cap = _CHART_CAPTIONS.get("Projection Quality", "")
                fig.add_annotation(
                    text=f"<span style='color:#666;font-size:10px'>{cap}</span>",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.14,
                    xanchor="center", yanchor="top",
                    showarrow=False,
                    align="center",
                )
                fig.update_layout(margin=dict(t=50, b=50, l=50, r=30))
                charts.append(("Projection Quality", fig))

        if getattr(report, "retrieval", None):
            r = report.retrieval
            if r:
                fig = self.retrieval_metrics_curve(
                    [x.k for x in r], [x.recall for x in r], [x.precision for x in r], [x.ndcg for x in r],
                )
                cap = _CHART_CAPTIONS.get("Retrieval Quality", "")
                fig.add_annotation(
                    text=f"<span style='color:#666;font-size:10px'>{cap}</span>",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.14,
                    xanchor="center", yanchor="top",
                    showarrow=False,
                    align="center",
                )
                fig.update_layout(margin=dict(t=50, b=50, l=50, r=30))
                charts.append(("Retrieval Quality", fig))

        return charts

    def _build_storytelling_figure(self, explanations, report) -> object:
        go = self._go
        fig = go.Figure()

        text = _format_storytelling(explanations)
        n_lines = text.count("<br>") + 1
        fig_height = max(120, 40 + n_lines * 22)

        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=0.04, y=0.95,
            xanchor="left", yanchor="top",
            showarrow=False,
            align="left",
            font=dict(size=11),
        )

        model_name = getattr(report, "model_name", "unnamed")
        fig.update_layout(
            title=dict(
                text=(
                    "<b>Analysis & Recommendations</b><br>"
                    f"<span style='font-size:11px;color:#555'>{model_name}</span>"
                ),
                x=0.5, xanchor="center",
            ),
            height=fig_height,
            margin=dict(t=50, b=15, l=30, r=30),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    def full_dashboard(self, report, interpretation=None) -> tuple:
        go = self._go

        fig_kpi = _build_kpi_figure(go, report)

        chart_list = self._build_chart_list(report)

        explanations = (
            getattr(interpretation, "explanations", None) or []
            if interpretation is not None
            else []
        )
        fig_storytelling = (
            self._build_storytelling_figure(explanations, report)
            if interpretation is not None
            else None
        )

        return fig_kpi, chart_list, fig_storytelling
