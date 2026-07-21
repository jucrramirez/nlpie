"""Tests for multi-model comparison charts (metric alignment & validation)."""

import math

import pytest
from nlpie.dashboard.comparison import _collect_metric_values, compare_and_plot_delta
from nlpie.metrics.quality import (
    ClusteringMetrics,
    EmbeddingQualityReport,
    GeometryMetrics,
    IntrinsicMetrics,
    ProjectionMetrics,
    RetrievalMetrics,
)


def _report(
    name: str,
    *,
    intrinsic: bool = True,
    clustering: bool = False,
    geometry: bool = True,
    projection: bool = False,
    retrieval: bool = False,
) -> EmbeddingQualityReport:
    r = EmbeddingQualityReport(model_name=name, n_samples=100, n_dims=64)
    if intrinsic:
        r.intrinsic = IntrinsicMetrics(mean=0.5, std=0.1, min=0.2, max=0.8)
    if clustering:
        r.clustering = ClusteringMetrics(
            ari=0.7, nmi=0.6, purity=0.8, silhouette=0.4, calinski_harabasz=100.0
        )
    if geometry:
        r.geometry = GeometryMetrics(
            effective_rank=32.0, mean_similarity=0.5, hubness_skewness=0.3, hubness_k=5
        )
    if projection:
        r.projection = [ProjectionMetrics(k=5, trustworthiness=0.9, continuity=0.85)]
    if retrieval:
        r.retrieval = [
            RetrievalMetrics(k=5, recall=0.6, precision=0.5, mrr=0.7, ndcg=0.65, coverage=0.4)
        ]
    return r


class TestCollectMetricValues:
    def test_aligned_when_all_sections_present(self):
        reports = [_report("a", clustering=True), _report("b", clustering=True)]
        model_names, metric_names, values = _collect_metric_values(reports)
        assert model_names == ["a", "b"]
        for row in values:
            assert len(row) == len(metric_names)
            assert all(v is not None for v in row)

    def test_mixed_sections_do_not_misalign(self):
        """Regression (B3): a model missing a section must get None placeholders,
        never shifted values."""
        reports = [
            _report("full", clustering=True, projection=True, retrieval=True),
            _report("sparse"),
        ]
        _, metric_names, values = _collect_metric_values(reports)

        full_row, sparse_row = values
        assert len(full_row) == len(sparse_row) == len(metric_names)

        ari_idx = metric_names.index("ARI")
        assert full_row[ari_idx] == 0.7
        assert sparse_row[ari_idx] is None

        # Values after the missing section must still land in their own columns.
        eff_idx = metric_names.index("Eff-Rank (frac)")
        assert math.isclose(full_row[eff_idx], 32.0 / 64.0)
        assert math.isclose(sparse_row[eff_idx], 32.0 / 64.0)

        ndcg_idx = metric_names.index("nDCG")
        assert sparse_row[ndcg_idx] is None
        assert math.isclose(full_row[ndcg_idx], 0.65)

    def test_effective_rank_is_normalized_to_unit_fraction(self):
        reports = [_report("a")]
        _, metric_names, values = _collect_metric_values(reports)
        idx = metric_names.index("Eff-Rank (frac)")
        assert math.isclose(values[0][idx], 0.5)

    def test_metric_order_is_stable(self):
        reports = [_report("b", clustering=True), _report("a", clustering=True)]
        _, metric_names_a, _ = _collect_metric_values(reports)
        _, metric_names_b, _ = _collect_metric_values(list(reversed(reports)))
        assert metric_names_a == metric_names_b


class TestCompareAndPlotDelta:
    def _models(self):
        emb_a = [[float(i + j) for j in range(4)] for i in range(20)]
        emb_b = [[float(i * 2 + j) for j in range(4)] for i in range(20)]
        return {"base": emb_a, "candidate": emb_b}

    def test_unknown_baseline_raises(self):
        """Regression (B5): a misspelled baseline must raise, not silently
        fall back to the first model."""
        with pytest.raises(ValueError, match="Unknown baseline"):
            compare_and_plot_delta(self._models(), baseline="bass", hubness_k=3)

    def test_valid_baseline_produces_zero_self_delta(self):
        pytest.importorskip("plotly")
        fig = compare_and_plot_delta(self._models(), baseline="base", hubness_k=3)
        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)
        # The baseline row must be all zeros (or None for missing sections).
        heatmap = fig.data[0]
        base_row = heatmap.z[list(heatmap.y).index("base")]
        assert all(v in (0.0, None) for v in base_row)
