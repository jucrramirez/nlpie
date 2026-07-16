"""Tests for retrieval and ranking metrics (TASK-006):
Recall@K, Precision@K, MRR, nDCG@K, Coverage@K.
"""

import math

import pytest

from nlpie._api import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    coverage_at_k,
)
from nlpie._errors import PreprocessingError


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        assert math.isclose(recall_at_k([0, 1, 2], [0, 1, 2], k=3), 1.0)

    def test_partial_recall(self):
        # Only doc 0 is in the top-3 retrieved; 3 relevant → recall = 1/3
        score = recall_at_k([0, 5, 6, 1, 2], [0, 1, 2], k=3)
        assert math.isclose(score, 1.0 / 3.0, rel_tol=1e-6)

    def test_zero_recall_when_no_hit(self):
        assert math.isclose(recall_at_k([5, 6, 7], [0, 1, 2], k=3), 0.0)

    def test_empty_relevant_returns_zero(self):
        assert math.isclose(recall_at_k([0, 1, 2], [], k=3), 0.0)

    def test_k_zero_raises(self):
        with pytest.raises(PreprocessingError):
            recall_at_k([0, 1, 2], [0], k=0)

    def test_k_larger_than_list_clips_to_list(self):
        # k=10 but only 3 items retrieved; should not error
        score = recall_at_k([0, 1, 2], [0, 1], k=10)
        assert math.isclose(score, 1.0)


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert math.isclose(precision_at_k([0, 1, 2], [0, 1, 2], k=3), 1.0)

    def test_half_precision(self):
        # 2 hits out of 4 → 0.5
        score = precision_at_k([0, 1, 5, 6], [0, 1, 2], k=4)
        assert math.isclose(score, 0.5, rel_tol=1e-6)

    def test_zero_precision(self):
        assert math.isclose(precision_at_k([5, 6, 7], [0, 1], k=3), 0.0)

    def test_empty_relevant_returns_zero(self):
        assert math.isclose(precision_at_k([0, 1], [], k=2), 0.0)

    def test_k_zero_raises(self):
        with pytest.raises(PreprocessingError):
            precision_at_k([0, 1], [0], k=0)


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_first_position_is_relevant(self):
        assert math.isclose(mean_reciprocal_rank([0, 1, 2], [0]), 1.0)

    def test_second_position(self):
        # Rank 2 → MRR = 0.5
        assert math.isclose(mean_reciprocal_rank([5, 0, 1], [0]), 0.5)

    def test_third_position(self):
        assert math.isclose(mean_reciprocal_rank([5, 6, 0], [0]), 1.0 / 3.0, rel_tol=1e-6)

    def test_no_hit_returns_zero(self):
        assert math.isclose(mean_reciprocal_rank([5, 6, 7], [0]), 0.0)

    def test_empty_relevant_returns_zero(self):
        assert math.isclose(mean_reciprocal_rank([0, 1], []), 0.0)

    def test_empty_retrieved_raises(self):
        with pytest.raises(PreprocessingError):
            mean_reciprocal_rank([], [0])


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_ndcg(self):
        assert math.isclose(ndcg_at_k([0, 1, 2], [0, 1, 2], k=3), 1.0)

    def test_single_relevant_at_first_rank(self):
        # DCG = IDCG = 1/log2(2) = 1 → nDCG = 1
        assert math.isclose(ndcg_at_k([0, 5, 6], [0], k=3), 1.0)

    def test_relevant_at_second_rank(self):
        # DCG = 1/log2(3) ≈ 0.6309; IDCG = 1/log2(2) = 1 → nDCG ≈ 0.6309
        score = ndcg_at_k([5, 0], [0], k=2)
        expected = 1.0 / math.log2(3)
        assert math.isclose(score, expected, rel_tol=1e-5)

    def test_no_hit_returns_zero(self):
        assert math.isclose(ndcg_at_k([5, 6, 7], [0, 1], k=3), 0.0)

    def test_empty_relevant_returns_zero(self):
        assert math.isclose(ndcg_at_k([0, 1], [], k=2), 0.0)

    def test_k_zero_raises(self):
        with pytest.raises(PreprocessingError):
            ndcg_at_k([0, 1], [0], k=0)


# ---------------------------------------------------------------------------
# Coverage@K
# ---------------------------------------------------------------------------

class TestCoverageAtK:
    def test_full_coverage(self):
        retrieved = [[0, 1], [2, 3]]
        relevant = [[0, 1], [2, 3]]
        assert math.isclose(coverage_at_k(retrieved, relevant, k=2), 1.0)

    def test_partial_coverage(self):
        # Only doc 0 covered; 4 total relevant → 0.25
        retrieved = [[0, 5], [6, 7]]
        relevant = [[0, 1], [2, 3]]
        assert math.isclose(coverage_at_k(retrieved, relevant, k=2), 0.25, rel_tol=1e-6)

    def test_zero_coverage(self):
        retrieved = [[5, 6], [7, 8]]
        relevant = [[0, 1], [2, 3]]
        assert math.isclose(coverage_at_k(retrieved, relevant, k=2), 0.0)

    def test_k_zero_raises(self):
        with pytest.raises(PreprocessingError):
            coverage_at_k([[0, 1]], [[0]], k=0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(PreprocessingError):
            coverage_at_k([[0, 1], [2, 3]], [[0]], k=2)

    def test_empty_lists_raises(self):
        with pytest.raises(PreprocessingError):
            coverage_at_k([], [], k=1)


# ---------------------------------------------------------------------------
# High-level RetrievalReport helper
# ---------------------------------------------------------------------------

class TestEvaluateRetrieval:
    def _import(self):
        from nlpie.metrics.retrieval import evaluate_retrieval
        return evaluate_retrieval

    def test_report_has_correct_k_values(self):
        evaluate_retrieval = self._import()
        retrieved = [[0, 1, 2], [0, 1, 2]]
        relevant = [[0, 1], [1, 2]]
        report = evaluate_retrieval(retrieved, relevant, k_values=[1, 2, 3])
        assert [s.k for s in report.scores] == [1, 2, 3]

    def test_perfect_retrieval_scores(self):
        evaluate_retrieval = self._import()
        retrieved = [[0, 1, 2], [3, 4, 5]]
        relevant = [[0, 1, 2], [3, 4, 5]]
        report = evaluate_retrieval(retrieved, relevant, k_values=[3])
        s = report.scores[0]
        assert math.isclose(s.recall, 1.0)
        assert math.isclose(s.precision, 1.0)
        assert math.isclose(s.ndcg, 1.0)
        assert math.isclose(s.coverage, 1.0)
        assert math.isclose(s.mrr, 1.0)

    def test_empty_k_values_raises(self):
        evaluate_retrieval = self._import()
        with pytest.raises(ValueError):
            evaluate_retrieval([[0]], [[0]], k_values=[])

    def test_mismatched_query_lists_raises(self):
        evaluate_retrieval = self._import()
        with pytest.raises(ValueError):
            evaluate_retrieval([[0, 1], [2, 3]], [[0]], k_values=[1])

    def test_str_report_is_non_empty(self):
        evaluate_retrieval = self._import()
        retrieved = [[0, 1, 2]]
        relevant = [[0, 1]]
        report = evaluate_retrieval(retrieved, relevant, k_values=[1, 2])
        text = str(report)
        assert "Retrieval Quality Report" in text
        assert "k=" in text
