import math

import pytest

from nlpie.backends.base import PlotBackend


class MockBackend(PlotBackend):
    def __init__(self):
        self.calls = []

    def hubness_histogram(self, counts, skewness, k):
        self.calls.append(("hubness_histogram", counts, skewness, k))
        return "hubness_fig"

    def similarity_distribution(self, values, mean, std):
        self.calls.append(("similarity_distribution", values, mean, std))
        return "sim_fig"

    def similarity_to_mean_chart(self, sims):
        self.calls.append(("similarity_to_mean_chart", sims))
        return "sim_mean_fig"

    def projection_quality_curve(self, k_values, trustworthiness, continuity):
        self.calls.append(("projection_quality_curve", k_values, trustworthiness, continuity))
        return "proj_fig"

    def retrieval_metrics_curve(self, k_values, recall, precision, ndcg):
        self.calls.append(("retrieval_metrics_curve", k_values, recall, precision, ndcg))
        return "ret_fig"

    def full_dashboard(self, report, interpretation=None):
        self.calls.append(("full_dashboard", report, interpretation))
        return "dash_fig"


class TestMockBackend:
    def test_hubness_histogram(self):
        backend = MockBackend()
        result = backend.hubness_histogram([5, 5, 5], 0.1, 5)
        assert result == "hubness_fig"
        assert backend.calls[0][0] == "hubness_histogram"

    def test_similarity_distribution(self):
        backend = MockBackend()
        result = backend.similarity_distribution([0.1, 0.2], 0.15, 0.05)
        assert result == "sim_fig"

    def test_similarity_to_mean_chart(self):
        backend = MockBackend()
        result = backend.similarity_to_mean_chart([0.5, 0.6, 0.7])
        assert result == "sim_mean_fig"

    def test_projection_quality_curve(self):
        backend = MockBackend()
        result = backend.projection_quality_curve([5, 10], [0.9, 0.8], [0.8, 0.9])
        assert result == "proj_fig"

    def test_retrieval_metrics_curve(self):
        backend = MockBackend()
        result = backend.retrieval_metrics_curve([1, 5], [0.5, 0.8], [0.4, 0.6], [0.5, 0.7])
        assert result == "ret_fig"


class TestDashboardFunctions:
    def test_plot_hubness_histogram_with_mock(self):
        mock = MockBackend()
        from nlpie.dashboard import plot_hubness_histogram
        result = plot_hubness_histogram([5, 5, 5], 0.1, 5, backend=mock)
        assert result == "hubness_fig"

    def test_plot_similarity_distribution_with_mock(self):
        mock = MockBackend()
        from nlpie.dashboard import plot_similarity_distribution
        result = plot_similarity_distribution([0.1, 0.2], 0.15, 0.05, backend=mock)
        assert result == "sim_fig"

    def test_plot_similarity_to_mean_with_mock(self):
        mock = MockBackend()
        from nlpie.dashboard import plot_similarity_to_mean
        result = plot_similarity_to_mean([0.5, 0.6], backend=mock)
        assert result == "sim_mean_fig"

    def test_plot_projection_quality_with_mock(self):
        mock = MockBackend()
        from nlpie.dashboard import plot_projection_quality
        result = plot_projection_quality([5, 10], [0.9, 0.8], [0.8, 0.9], backend=mock)
        assert result == "proj_fig"

    def test_plot_retrieval_metrics_with_mock(self):
        mock = MockBackend()
        from nlpie.dashboard import plot_retrieval_metrics
        result = plot_retrieval_metrics([1, 5], [0.5, 0.8], [0.4, 0.6], [0.5, 0.7], backend=mock)
        assert result == "ret_fig"

    def test_plot_hubness_default_backend(self):
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")
        from nlpie.dashboard import plot_hubness_histogram
        fig = plot_hubness_histogram([5, 5, 5], 0.0, 5)
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_plotly_backend_raises_without_plotly(self):
        import sys as _sys
        import builtins

        cached = {k: v for k, v in _sys.modules.items() if k.startswith("plotly")}
        for k in cached:
            del _sys.modules[k]

        orig_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            top = name.split(".")[0]
            if top == "plotly":
                raise ImportError("No module named plotly")
            return orig_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            from nlpie.backends.plotly import PlotlyBackend
            with pytest.raises(ImportError, match="Plotly is required"):
                PlotlyBackend()
        finally:
            builtins.__import__ = orig_import
            _sys.modules.update(cached)


class TestFullDashboard:
    def test_full_dashboard_with_mock_backend(self):
        mock = MockBackend()
        report = object()
        result = mock.full_dashboard(report)
        assert result == "dash_fig"
        assert mock.calls[0] == ("full_dashboard", report, None)

    def test_plot_quality_report_with_mock(self):
        mock = MockBackend()
        from nlpie.dashboard import plot_quality_report
        report = object()
        result = plot_quality_report(report, backend=mock)
        assert result == "dash_fig"
        assert mock.calls[0] == ("full_dashboard", report, None)

    def test_plot_quality_report_default_backend(self):
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")
        from nlpie.dashboard import plot_quality_report
        from nlpie.metrics.quality import EmbeddingQualityReport, IntrinsicMetrics
        report = EmbeddingQualityReport(
            model_name="test",
            n_samples=100,
            n_dims=64,
            intrinsic=IntrinsicMetrics(mean=0.5, std=0.1, min=0.2, max=0.8),
        )
        result = plot_quality_report(report)
        import plotly.graph_objects as go
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig_kpi, chart_list, fig_story = result
        assert isinstance(fig_kpi, go.Figure)
        assert isinstance(chart_list, list)
        for _, fig in chart_list:
            assert isinstance(fig, go.Figure)
        assert fig_story is None

    def test_full_dashboard_empty_report(self):
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")
        from nlpie.backends.plotly import PlotlyBackend
        from nlpie.metrics.quality import EmbeddingQualityReport
        report = EmbeddingQualityReport()
        backend = PlotlyBackend()
        result = backend.full_dashboard(report)
        import plotly.graph_objects as go
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig_kpi, chart_list, fig_story = result
        assert isinstance(fig_kpi, go.Figure)
        assert isinstance(chart_list, list)
        assert fig_story is None


class TestNewPlotTypes:
    @pytest.fixture(autouse=True)
    def _check_plotly(self):
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")

    def test_similarity_heatmap(self):
        from nlpie.dashboard import plot_similarity_heatmap
        matrix = [[1.0, 0.5, 0.0], [0.5, 1.0, -0.2], [0.0, -0.2, 1.0]]
        fig = plot_similarity_heatmap(matrix)
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_similarity_heatmap_with_labels(self):
        from nlpie.dashboard import plot_similarity_heatmap
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        fig = plot_similarity_heatmap(matrix, labels=[0, 1])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_hubness_bar(self):
        from nlpie.dashboard import plot_hubness_bar
        fig = plot_hubness_bar([5, 10, 3, 0, 7], top_n=5)
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_eigenvalue_scree(self):
        from nlpie.dashboard import plot_eigenvalue_scree
        fig = plot_eigenvalue_scree([3.0, 1.5, 0.8, 0.4, 0.2])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_silhouette_plot(self):
        from nlpie.dashboard import plot_silhouette
        fig = plot_silhouette([0.8, 0.6, 0.3, -0.1, 0.5], [0, 0, 1, 1, 2])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap(self):
        from nlpie.dashboard import plot_correlation_heatmap
        corr = [[1.0, 0.3], [0.3, 1.0]]
        fig = plot_correlation_heatmap(corr, ["ARI", "NMI"])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_embedding_scatter(self):
        from nlpie.dashboard import plot_embedding_scatter
        fig = plot_embedding_scatter([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], labels=[0, 1, 0])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_comparison_radar(self):
        from nlpie.dashboard import plot_comparison_radar
        fig = plot_comparison_radar(["A", "B"], ["R@K", "nDCG"], [[0.8, 0.7], [0.9, 0.85]])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_comparison_grouped_bar(self):
        from nlpie.dashboard import plot_comparison_grouped_bar
        fig = plot_comparison_grouped_bar(["A", "B"], ["R@K", "nDCG"], [[0.8, 0.7], [0.9, 0.85]])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_comparison_delta(self):
        from nlpie.dashboard import plot_comparison_delta
        fig = plot_comparison_delta(["A", "B"], ["R@K", "nDCG"], [[0.0, 0.1], [-0.05, 0.0]])
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_dashboard_imports_from_package(self):
        from nlpie.dashboard import plot_hubness_histogram, plot_similarity_distribution, plot_projection_quality, plot_quality_report, plot_similarity_heatmap, plot_hubness_bar, plot_eigenvalue_scree, plot_silhouette, plot_correlation_heatmap, plot_embedding_scatter, plot_comparison_radar, plot_comparison_grouped_bar, plot_comparison_delta
        mock = MockBackend()
        result = plot_hubness_histogram([5, 5, 5], 0.1, 5, backend=mock)
        assert result == "hubness_fig"


class TestDashboardBuilder:
    def test_builder_with_sections(self):
        from nlpie.dashboard import DashboardBuilder
        from nlpie.metrics.quality import EmbeddingQualityReport, IntrinsicMetrics, GeometryMetrics
        report = EmbeddingQualityReport(
            model_name="test",
            n_samples=100, n_dims=64,
            intrinsic=IntrinsicMetrics(mean=0.5, std=0.1, min=0.2, max=0.8),
            geometry=GeometryMetrics(effective_rank=32, mean_similarity=0.5, hubness_skewness=0.3, hubness_k=5),
        )
        report.hubness_counts = [5] * 100
        figs = (
            DashboardBuilder(report)
            .add_hubness_histogram()
            .add_similarity_distribution()
            .build()
        )
        import plotly.graph_objects as go
        assert isinstance(figs, list)
        assert len(figs) == 2
        assert all(isinstance(f, go.Figure) for f in figs)

    def test_builder_fallback_full_dashboard(self):
        from nlpie.dashboard import DashboardBuilder
        from nlpie.metrics.quality import EmbeddingQualityReport
        report = EmbeddingQualityReport(model_name="test", n_samples=100, n_dims=64)
        figs = DashboardBuilder(report).build()
        import plotly.graph_objects as go
        assert isinstance(figs, list)
        for f in figs:
            assert isinstance(f, go.Figure)
