"""Tests for projection quality metrics (TASK-005): trustworthiness and continuity."""

import math

import pytest
from nlpie._api import continuity, projection_quality, trustworthiness
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

    def test_rejects_degenerate_k(self):
        """k values with a non-positive normalisation denominator must error.

        Regression test: n=2, k=1 and n=5, k=3 both make 2n - 3k - 1 <= 0,
        which previously produced NaN or scores outside [0, 1].
        """
        with pytest.raises(PreprocessingError):
            trustworthiness([[1.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]], k=1)
        emb = _identity_5x2()
        with pytest.raises(PreprocessingError):
            trustworthiness(emb, emb, k=3)

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
        """Calling without explicit k should not raise when k=10 is in-domain."""
        # Need 2n - 3*10 - 1 > 0 → n >= 16.
        emb = [[float(i), 0.0] for i in range(31)]
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

    def test_rejects_degenerate_k(self):
        emb = _identity_5x2()
        with pytest.raises(PreprocessingError):
            continuity(emb, emb, k=3)

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
        emb = [[float(i), 0.0] for i in range(31)]
        score = continuity(emb, emb)
        assert math.isclose(score, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# One-pass projection_quality
# ---------------------------------------------------------------------------


class TestProjectionQuality:
    def test_matches_individual_calls(self):
        high = [[float(i), 0.0] for i in range(10)]
        low = [[float(9 - i), 0.0] for i in range(10)]
        results = projection_quality(high, low, k_values=[2, 4, 5])
        assert [k for k, _, _ in results] == [2, 4, 5]
        for k, t, c in results:
            assert math.isclose(t, trustworthiness(high, low, k=k), abs_tol=1e-6)
            assert math.isclose(c, continuity(high, low, k=k), abs_tol=1e-6)

    def test_perfect_projection(self):
        emb = [[float(i), 0.0] for i in range(10)]
        results = projection_quality(emb, emb, k_values=[2, 4])
        for _, t, c in results:
            assert math.isclose(t, 1.0, abs_tol=1e-5)
            assert math.isclose(c, 1.0, abs_tol=1e-5)

    def test_empty_k_values_raises(self):
        emb = [[float(i), 0.0] for i in range(10)]
        with pytest.raises(PreprocessingError):
            projection_quality(emb, emb, k_values=[])

    def test_invalid_k_raises(self):
        emb = _identity_5x2()
        with pytest.raises(PreprocessingError):
            projection_quality(emb, emb, k_values=[1, 3])

    def test_mismatched_rows_raises(self):
        high = [[float(i), 0.0] for i in range(5)]
        low = [[float(i), 0.0] for i in range(4)]
        with pytest.raises(PreprocessingError):
            projection_quality(high, low, k_values=[2])
