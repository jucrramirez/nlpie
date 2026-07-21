import numpy as np
import pytest
from nlpie._api import (
    cosine_similarity_matrix,
    pairwise_cosine_stats,
    pearson_correlation,
    spearman_correlation,
)
from nlpie._similarity import reconstruct_similarity_matrix


def test_cosine_similarity_matrix():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    sim = cosine_similarity_matrix(x)
    assert np.allclose(sim[0][1], 0.0, atol=1e-4)
    assert np.allclose(sim[0][2], 0.7071, atol=1e-4)
    assert np.allclose(sim[1][2], 0.7071, atol=1e-4)
    assert np.allclose(sim[0][0], 1.0, atol=1e-4)


def test_pairwise_cosine_stats_triangle_size_and_order():
    x = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7071, 0.7071],
        ]
    )
    triangle, mean, std, min_val, max_val = pairwise_cosine_stats(x)
    # n=3 → exactly 3 pairs: (0,1), (0,2), (1,2) in row-major order.
    assert len(triangle) == 3
    assert np.allclose(triangle[0], 0.0, atol=1e-4)
    assert np.allclose(triangle[1], 0.7071, atol=1e-4)
    assert np.allclose(triangle[2], 0.7071, atol=1e-4)
    assert np.isclose(min_val, min(triangle))
    assert np.isclose(max_val, max(triangle))
    assert np.isclose(mean, sum(triangle) / len(triangle))


def test_pairwise_cosine_stats_matches_full_matrix():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((30, 8)).astype(np.float32)
    triangle, mean, std, min_val, max_val = pairwise_cosine_stats(x)
    full = np.asarray(cosine_similarity_matrix(x))
    n = full.shape[0]
    expected = [full[i, j] for i in range(n) for j in range(i + 1, n)]
    assert np.allclose(triangle, expected, atol=1e-5)


def test_pairwise_cosine_stats_reconstruct_round_trip():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((12, 5)).astype(np.float32)
    triangle, *_ = pairwise_cosine_stats(x)
    rebuilt = reconstruct_similarity_matrix(triangle, n=12)
    full = np.asarray(cosine_similarity_matrix(x))
    assert np.allclose(rebuilt, full, atol=1e-5)


def test_reconstruct_similarity_matrix_validates_length():
    with pytest.raises(ValueError, match="upper triangle"):
        reconstruct_similarity_matrix([0.1, 0.2], n=3)


def test_pearson_correlation():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    assert np.isclose(pearson_correlation(x, y), 1.0)

    y2 = [-1.0, -2.0, -3.0, -4.0]
    assert np.isclose(pearson_correlation(x, y2), -1.0)

    # test 0 variance
    y3 = [1.0, 1.0, 1.0, 1.0]
    assert np.isclose(pearson_correlation(x, y3), 0.0)


def test_spearman_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 3.0, 1.0, 5.0, 4.0]
    # Ranks:
    # x: 1, 2, 3, 4, 5
    # y: 2, 3, 1, 5, 4
    # diff: -1, -1, 2, -1, 1
    # diff^2: 1, 1, 4, 1, 1 -> sum = 8
    # spearman = 1 - (6*8) / (5*(25-1)) = 1 - 48/120 = 1 - 0.4 = 0.6
    assert np.isclose(spearman_correlation(x, y), 0.6)


def test_spearman_correlation_ties():
    x = [1.0, 2.0, 2.0, 4.0]
    # Ranks: 1, 2.5, 2.5, 4
    y = [1.0, 2.0, 3.0, 4.0]
    # Ranks: 1, 2, 3, 4
    corr = spearman_correlation(x, y)
    assert corr > 0.8
