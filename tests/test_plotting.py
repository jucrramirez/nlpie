import sys
import math

import pytest

sys.path.insert(0, "python")

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
