"""Tests for projection quality metrics (TASK-005): trustworthiness and continuity."""

import math

import pytest

from nlpie._api import trustworthiness, continuity
from nlpie._errors import PreprocessingError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity_5x2():
    """5 points on a 1-D line; the 'projection' is the same matrix."""
    return [[float(i), 0.0] for i in range(5)]


# ---------------------------------------------------------------------------
# Trustworthiness
# ---------------------------------------------------------------------------

class TestTrustworthiness:
    def test_perfect_projection_returns_one(self):
        emb = _identity_5x2()
        score = trustworthiness(emb, emb, k=2)
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_score_in_unit_interval(self):
        high = [[float(i), 0.0] for i in range(10)]
        # Shuffle to create a deliberately bad 'projection'.
        low = [[float(9 - i), 0.0] for i in range(10)]
        score = trustworthiness(high, low, k=3)
        assert 0.0 <= score <= 1.0

    def test_rejects_k_zero(self):
        emb = _identity_5x2()
        with pytest.raises(PreprocessingError):
            trustworthiness(emb, emb, k=0)

    def test_rejects_k_equal_to_n(self):
        emb = _identity_5x2()
        with pytest.raises(PreprocessingError):
            trustworthiness(emb, emb, k=5)

    def test_rejects_mismatched_row_counts(self):
        high = [[float(i), 0.0] for i in range(5)]
        low = [[float(i), 0.0] for i in range(4)]
        with pytest.raises(PreprocessingError):
            trustworthiness(high, low, k=2)

    def test_accepts_numpy_arrays(self):
        pytest.importorskip("numpy")
        import numpy as np

        emb = np.array(_identity_5x2(), dtype=np.float32)
        score = trustworthiness(emb, emb, k=2)
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_default_k_is_10(self):
        """Calling without explicit k should not raise for n > 10."""
        emb = [[float(i), 0.0] for i in range(15)]
        score = trustworthiness(emb, emb)
        assert math.isclose(score, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------

class TestContinuity:
    def test_perfect_projection_returns_one(self):
        emb = _identity_5x2()
        score = continuity(emb, emb, k=2)
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_score_in_unit_interval(self):
        high = [[float(i), 0.0] for i in range(10)]
        low = [[float(9 - i), 0.0] for i in range(10)]
        score = continuity(high, low, k=3)
        assert 0.0 <= score <= 1.0

    def test_rejects_k_zero(self):
        emb = _identity_5x2()
        with pytest.raises(PreprocessingError):
            continuity(emb, emb, k=0)

    def test_rejects_mismatched_row_counts(self):
        high = [[float(i), 0.0] for i in range(5)]
        low = [[float(i), 0.0] for i in range(3)]
        with pytest.raises(PreprocessingError):
            continuity(high, low, k=2)

    def test_accepts_numpy_arrays(self):
        pytest.importorskip("numpy")
        import numpy as np

        emb = np.array(_identity_5x2(), dtype=np.float32)
        score = continuity(emb, emb, k=2)
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_default_k_is_10(self):
        emb = [[float(i), 0.0] for i in range(15)]
        score = continuity(emb, emb)
        assert math.isclose(score, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Projection report helper
# ---------------------------------------------------------------------------

class TestEvaluateProjection:
    def test_report_contains_all_k_values(self):
        from nlpie.metrics.projection import evaluate_projection

        emb = [[float(i), 0.0] for i in range(15)]
        report = evaluate_projection(emb, emb, k_values=[2, 5, 10])
        assert len(report.scores) == 3
        assert [s.k for s in report.scores] == [2, 5, 10]

    def test_perfect_projection_mean_scores(self):
        from nlpie.metrics.projection import evaluate_projection

        emb = [[float(i), 0.0] for i in range(15)]
        report = evaluate_projection(emb, emb, k_values=[2, 5])
        assert math.isclose(report.mean_trustworthiness, 1.0, abs_tol=1e-5)
        assert math.isclose(report.mean_continuity, 1.0, abs_tol=1e-5)

    def test_empty_k_values_raises(self):
        from nlpie.metrics.projection import evaluate_projection

        emb = [[float(i), 0.0] for i in range(15)]
        with pytest.raises(ValueError):
            evaluate_projection(emb, emb, k_values=[])
