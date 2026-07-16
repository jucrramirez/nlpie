"""Tests for the EmbeddingQualityReport and model comparison (TASK-007)."""

import math

import pytest

from nlpie.metrics.quality import (
    EmbeddingQualityReport,
    IntrinsicMetrics,
    ClusteringMetrics,
    GeometryMetrics,
    ProjectionMetrics,
    RetrievalMetrics,
    evaluate_embedding_quality,
    compare_models,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n: int = 30, d: int = 4) -> list[list[float]]:
    """Generates deterministic embeddings spread along the first axis."""
    return [[float(i + j) for j in range(d)] for i in range(n)]


def _make_labels(n: int = 30, n_classes: int = 3) -> list[int]:
    """Generates cyclic labels: 0, 1, 2, 0, 1, 2, ..."""
    return [i % n_classes for i in range(n)]


# ---------------------------------------------------------------------------
# EmbeddingQualityReport construction
# ---------------------------------------------------------------------------

class TestEmbeddingQualityReport:
    def test_str_contains_model_name(self):
        report = EmbeddingQualityReport(model_name="test-model", n_samples=10, n_dims=4)
        text = str(report)
        assert "test-model" in text

    def test_str_contains_sample_count(self):
        report = EmbeddingQualityReport(n_samples=42, n_dims=8)
        text = str(report)
        assert "42" in text
        assert "8" in text


# ---------------------------------------------------------------------------
# evaluate_embedding_quality — intrinsic & geometry only
# ---------------------------------------------------------------------------

class TestEvaluateIntrinsicAndGeometry:
    def test_intrinsic_metrics_populated(self):
        emb = _make_embeddings(20, 4)
        report, _ = evaluate_embedding_quality(emb, hubness_k=3)
        assert report.intrinsic is not None
        assert report.intrinsic.mean != 0.0
        assert report.intrinsic.std >= 0.0
        assert report.intrinsic.min <= report.intrinsic.max

    def test_geometry_metrics_populated(self):
        emb = _make_embeddings(20, 4)
        report, _ = evaluate_embedding_quality(emb, hubness_k=3)
        assert report.geometry is not None
        assert report.geometry.effective_rank > 0.0
        assert report.geometry.hubness_k == 3

    def test_clustering_not_populated_without_labels(self):
        emb = _make_embeddings(20, 4)
        report, _ = evaluate_embedding_quality(emb, hubness_k=3)
        assert report.clustering is None

    def test_projection_not_populated_without_low_dim(self):
        emb = _make_embeddings(20, 4)
        report, _ = evaluate_embedding_quality(emb, hubness_k=3)
        assert report.projection == []

    def test_retrieval_not_populated_without_data(self):
        emb = _make_embeddings(20, 4)
        report, _ = evaluate_embedding_quality(emb, hubness_k=3)
        assert report.retrieval == []


# ---------------------------------------------------------------------------
# evaluate_embedding_quality — with clustering labels
# ---------------------------------------------------------------------------

class TestEvaluateWithClustering:
    def test_clustering_populated_with_labels(self):
        emb = _make_embeddings(30, 4)
        labels = _make_labels(30, 3)
        report, _ = evaluate_embedding_quality(emb, labels=labels, hubness_k=3)
        assert report.clustering is not None
        assert math.isclose(report.clustering.ari, 1.0, abs_tol=1e-6)
        assert math.isclose(report.clustering.nmi, 1.0, abs_tol=1e-6)
        assert math.isclose(report.clustering.purity, 1.0, abs_tol=1e-6)

    def test_clustering_with_pred_labels(self):
        emb = _make_embeddings(30, 4)
        labels = _make_labels(30, 3)
        pred = _make_labels(30, 2)
        report, _ = evaluate_embedding_quality(emb, labels=labels, labels_pred=pred, hubness_k=3)
        assert report.clustering is not None
        assert report.clustering.ari < 1.0


# ---------------------------------------------------------------------------
# evaluate_embedding_quality — with projection
# ---------------------------------------------------------------------------

class TestEvaluateWithProjection:
    def test_projection_populated(self):
        emb = _make_embeddings(20, 4)
        low = [[float(i), float(i + 1)] for i in range(20)]
        report, _ = evaluate_embedding_quality(
            emb,
            low_dim=low,
            projection_k_values=[2, 5],
            hubness_k=3,
        )
        assert len(report.projection) == 2
        assert report.projection[0].k == 2
        assert report.projection[1].k == 5

    def test_perfect_projection_scores(self):
        emb = [[float(i), 0.0] for i in range(15)]
        report, _ = evaluate_embedding_quality(
            emb,
            low_dim=emb,
            projection_k_values=[2],
            hubness_k=3,
        )
        assert math.isclose(report.projection[0].trustworthiness, 1.0, abs_tol=1e-4)
        assert math.isclose(report.projection[0].continuity, 1.0, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# evaluate_embedding_quality — with retrieval
# ---------------------------------------------------------------------------

class TestEvaluateWithRetrieval:
    def test_retrieval_populated(self):
        emb = _make_embeddings(10, 4)
        retrieved = [[0, 1, 2], [3, 4, 5]]
        relevant = [[0, 1], [3, 4]]
        report, _ = evaluate_embedding_quality(
            emb,
            retrieved=retrieved,
            relevant=relevant,
            retrieval_k_values=[1, 2],
            hubness_k=3,
        )
        assert len(report.retrieval) == 2

    def test_perfect_retrieval_scores(self):
        emb = _make_embeddings(10, 4)
        retrieved = [[0, 1, 2], [3, 4, 5]]
        relevant = [[0, 1, 2], [3, 4, 5]]
        report, _ = evaluate_embedding_quality(
            emb,
            retrieved=retrieved,
            relevant=relevant,
            retrieval_k_values=[3],
            hubness_k=3,
        )
        s = report.retrieval[0]
        assert math.isclose(s.recall, 1.0)
        assert math.isclose(s.precision, 1.0)
        assert math.isclose(s.ndcg, 1.0)
        assert math.isclose(s.coverage, 1.0)
        assert math.isclose(s.mrr, 1.0)

    def test_retrieval_requires_both_lists(self):
        emb = _make_embeddings(10, 4)
        with pytest.raises(ValueError, match="Both"):
            evaluate_embedding_quality(emb, retrieved=[[0, 1]], hubness_k=3)
        with pytest.raises(ValueError, match="Both"):
            evaluate_embedding_quality(emb, relevant=[[0, 1]], hubness_k=3)


# ---------------------------------------------------------------------------
# Full report with all sections
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_all_sections_populated(self):
        emb = _make_embeddings(20, 4)
        labels = _make_labels(20, 3)
        low = [[float(i), 0.0] for i in range(20)]
        retrieved = [[0, 1, 2], [3, 4, 5]]
        relevant = [[0, 1], [3, 4]]

        report, _ = evaluate_embedding_quality(
            emb,
            labels=labels,
            hubness_k=3,
            low_dim=low,
            projection_k_values=[2],
            retrieved=retrieved,
            relevant=relevant,
            retrieval_k_values=[2],
            model_name="full-test",
        )
        assert report.intrinsic is not None
        assert report.clustering is not None
        assert report.geometry is not None
        assert len(report.projection) == 1
        assert len(report.retrieval) == 1
        assert report.model_name == "full-test"

    def test_str_representation_non_empty(self):
        emb = _make_embeddings(20, 4)
        labels = _make_labels(20, 3)
        report, _ = evaluate_embedding_quality(emb, labels=labels, hubness_k=3)
        text = str(report)
        assert "Embedding Quality Report" in text
        assert "Intrinsic" in text
        assert "Clustering" in text
        assert "Geometry" in text


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_one_report_per_model(self):
        emb_a = _make_embeddings(20, 4)
        emb_b = [[float(i * 2 + j) for j in range(4)] for i in range(20)]
        reports, _ = compare_models(
            {"model_a": emb_a, "model_b": emb_b},
            hubness_k=3,
        )
        assert len(reports) == 2
        assert reports[0].model_name == "model_a"
        assert reports[1].model_name == "model_b"

    def test_shared_labels_applied_to_all(self):
        emb_a = _make_embeddings(20, 4)
        emb_b = _make_embeddings(20, 4)
        labels = _make_labels(20, 3)
        reports, _ = compare_models(
            {"a": emb_a, "b": emb_b},
            labels=labels,
            hubness_k=3,
        )
        for r in reports:
            assert r.clustering is not None

    def test_with_per_model_projections(self):
        emb_a = [[float(i), 0.0] for i in range(15)]
        emb_b = [[float(i), 0.0] for i in range(15)]
        low_a = emb_a
        low_b = [[float(i % 3), float(i % 5)] for i in range(15)]

        reports, _ = compare_models(
            {"a": emb_a, "b": emb_b},
            low_dims={"a": low_a, "b": low_b},
            projection_k_values=[2],
            hubness_k=3,
        )
        assert math.isclose(reports[0].projection[0].trustworthiness, 1.0, abs_tol=1e-4)
        assert reports[1].projection[0].trustworthiness < 1.0

    def test_with_per_model_retrieval(self):
        emb_a = _make_embeddings(10, 4)
        emb_b = _make_embeddings(10, 4)
        relevant = [[0, 1], [3, 4]]
        retrieved_a = [[0, 1, 2], [3, 4, 5]]
        retrieved_b = [[5, 6, 7], [8, 9, 0]]

        reports, _ = compare_models(
            {"a": emb_a, "b": emb_b},
            retrieved={"a": retrieved_a, "b": retrieved_b},
            relevant=relevant,
            retrieval_k_values=[2],
            hubness_k=3,
        )
        assert reports[0].retrieval[0].recall > reports[1].retrieval[0].recall
