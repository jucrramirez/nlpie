"""Tests for the interpretation providers, registry, and storytelling data."""

from nlpie import thresholds
from nlpie.interpret import (
    ClusteringExplanationProvider,
    Explanation,
    ExplanationRegistry,
    GeometryExplanationProvider,
    IntrinsicExplanationProvider,
    ProjectionExplanationProvider,
    RetrievalExplanationProvider,
    build_story,
)
from nlpie.metrics.quality import (
    ClusteringMetrics,
    EmbeddingQualityReport,
    GeometryMetrics,
    IntrinsicMetrics,
    ProjectionMetrics,
    RetrievalMetrics,
)


def _report() -> EmbeddingQualityReport:
    return EmbeddingQualityReport(model_name="t", n_samples=100, n_dims=64)


class TestIntrinsicProvider:
    def test_high_mean_is_critical(self):
        report = _report()
        report.intrinsic = IntrinsicMetrics(mean=0.9, std=0.1, min=0.8, max=0.99)
        exps = IntrinsicExplanationProvider().explain(report)
        assert len(exps) == 1
        assert exps[0].severity == "critical"
        assert exps[0].metric == "intrinsic"

    def test_moderate_mean_is_warning(self):
        report = _report()
        report.intrinsic = IntrinsicMetrics(mean=0.6, std=0.1, min=0.4, max=0.8)
        assert IntrinsicExplanationProvider().explain(report)[0].severity == "warning"

    def test_typical_mean_is_info(self):
        report = _report()
        report.intrinsic = IntrinsicMetrics(mean=0.3, std=0.1, min=0.0, max=0.6)
        assert IntrinsicExplanationProvider().explain(report)[0].severity == "info"

    def test_missing_section_returns_empty(self):
        assert IntrinsicExplanationProvider().explain(_report()) == []


class TestClusteringProvider:
    def test_healthy_clustering_is_info(self):
        report = _report()
        report.clustering = ClusteringMetrics(
            ari=0.9, nmi=0.9, purity=0.95, silhouette=0.6, calinski_harabasz=300.0
        )
        exps = ClusteringExplanationProvider().explain(report)
        assert exps[0].severity == "info"

    def test_low_ari_is_critical(self):
        report = _report()
        report.clustering = ClusteringMetrics(
            ari=0.1, nmi=0.4, purity=0.5, silhouette=0.3, calinski_harabasz=50.0
        )
        exps = ClusteringExplanationProvider().explain(report)
        assert exps[0].severity == "critical"
        assert "ARI" in exps[0].summary

    def test_low_silhouette_is_critical(self):
        report = _report()
        report.clustering = ClusteringMetrics(
            ari=0.6, nmi=0.6, purity=0.7, silhouette=0.05, calinski_harabasz=50.0
        )
        assert ClusteringExplanationProvider().explain(report)[0].severity == "critical"


class TestGeometryProvider:
    def test_very_low_effective_rank_is_critical(self):
        report = _report()
        report.geometry = GeometryMetrics(
            effective_rank=10.0, mean_similarity=0.2, hubness_skewness=0.1, hubness_k=5
        )
        exps = GeometryExplanationProvider().explain(report)
        by_metric = {e.metric: e for e in exps}
        assert by_metric["effective_rank"].severity == "critical"

    def test_similarity_to_mean_is_explained(self):
        """Regression: the provider registers 'similarity_to_mean' — it must
        actually produce an explanation for it."""
        report = _report()
        report.geometry = GeometryMetrics(
            effective_rank=60.0, mean_similarity=0.95, hubness_skewness=0.1, hubness_k=5
        )
        exps = GeometryExplanationProvider().explain(report)
        by_metric = {e.metric: e for e in exps}
        assert "similarity_to_mean" in by_metric
        assert by_metric["similarity_to_mean"].severity == "critical"

    def test_missing_section_returns_empty(self):
        assert GeometryExplanationProvider().explain(_report()) == []


class TestProjectionProvider:
    def test_poor_projection_is_critical(self):
        report = _report()
        report.projection = [ProjectionMetrics(k=5, trustworthiness=0.4, continuity=0.9)]
        assert ProjectionExplanationProvider().explain(report)[0].severity == "critical"

    def test_moderate_projection_is_warning(self):
        report = _report()
        report.projection = [ProjectionMetrics(k=5, trustworthiness=0.7, continuity=0.75)]
        assert ProjectionExplanationProvider().explain(report)[0].severity == "warning"

    def test_good_projection_is_info(self):
        report = _report()
        report.projection = [ProjectionMetrics(k=5, trustworthiness=0.95, continuity=0.9)]
        assert ProjectionExplanationProvider().explain(report)[0].severity == "info"

    def test_empty_projection_returns_empty(self):
        assert ProjectionExplanationProvider().explain(_report()) == []


class TestRetrievalProvider:
    def test_low_recall_is_critical(self):
        report = _report()
        report.retrieval = [
            RetrievalMetrics(k=5, recall=0.2, precision=0.3, mrr=0.4, ndcg=0.3, coverage=0.2)
        ]
        assert RetrievalExplanationProvider().explain(report)[0].severity == "critical"

    def test_moderate_recall_is_warning(self):
        report = _report()
        report.retrieval = [
            RetrievalMetrics(k=5, recall=0.5, precision=0.5, mrr=0.6, ndcg=0.5, coverage=0.4)
        ]
        assert RetrievalExplanationProvider().explain(report)[0].severity == "warning"

    def test_good_recall_is_info(self):
        report = _report()
        report.retrieval = [
            RetrievalMetrics(k=5, recall=0.9, precision=0.8, mrr=0.9, ndcg=0.85, coverage=0.7)
        ]
        assert RetrievalExplanationProvider().explain(report)[0].severity == "info"


class TestThresholdConsistency:
    """Providers and KPI cards must classify the same value identically."""

    def test_kpi_and_providers_share_thresholds(self):
        from nlpie._kpi import kpi_cards_data

        report = _report()
        report.intrinsic = IntrinsicMetrics(mean=0.9, std=0.1, min=0.8, max=0.99)
        report.geometry = GeometryMetrics(
            effective_rank=10.0, mean_similarity=0.9, hubness_skewness=1.5, hubness_k=5
        )
        report.hubness_counts = [50] + [0] * 99

        cards = {c.label: c.severity for c in kpi_cards_data(report)}
        explanations = {
            e.metric: e.severity for e in ExplanationRegistry.explain_all(report).explanations
        }

        assert cards["Mean"] == explanations["intrinsic"] == "critical"
        assert cards["Eff-Rank"] == explanations["effective_rank"] == "critical"
        assert cards["Hubness"] == explanations["hubness"] == "critical"

    def test_classify_helpers(self):
        assert thresholds.classify_higher_is_worse(1.5, 1.0, 0.5) == "critical"
        assert thresholds.classify_higher_is_worse(0.7, 1.0, 0.5) == "warning"
        assert thresholds.classify_higher_is_worse(0.2, 1.0, 0.5) == "info"
        assert thresholds.classify_lower_is_worse(0.2, 0.3, 0.7) == "critical"
        assert thresholds.classify_lower_is_worse(0.5, 0.3, 0.7) == "warning"
        assert thresholds.classify_lower_is_worse(0.9, 0.3, 0.7) == "info"
        assert thresholds.worst_severity(["info", "warning", "info"]) == "warning"
        assert thresholds.worst_severity([]) == "info"


class TestBuildStory:
    def _exp(self, severity: str, metric: str = "m", rec: str = "fix it") -> Explanation:
        return Explanation(
            metric=metric, severity=severity, summary="s", detail="d", recommendation=rec
        )

    def test_empty_is_healthy(self):
        story = build_story([])
        assert story.is_healthy
        assert story.headline_severity == "ok"
        assert story.groups == []

    def test_counts_and_headline(self):
        story = build_story([self._exp("critical"), self._exp("warning"), self._exp("info")])
        assert story.n_critical == 1
        assert story.n_warning == 1
        assert story.n_info == 1
        assert story.headline_severity == "critical"
        assert "1 critical" in story.headline

    def test_groups_ordered_by_severity(self):
        story = build_story([self._exp("info"), self._exp("critical")])
        assert [g.severity for g in story.groups] == ["critical", "info"]

    def test_recommendations_collected(self):
        story = build_story([self._exp("warning", rec="do A"), self._exp("info", rec="do B")])
        assert story.recommendations == ["do A", "do B"]
