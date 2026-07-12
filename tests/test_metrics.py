import pytest
import numpy as np
from nlpie._api import cosine_similarity_matrix, pearson_correlation, spearman_correlation
from nlpie._errors import PreprocessingError

def test_cosine_similarity_matrix():
    x = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7071, 0.7071]
    ])
    sim = cosine_similarity_matrix(x)
    assert np.allclose(sim[0][1], 0.0, atol=1e-4)
    assert np.allclose(sim[0][2], 0.7071, atol=1e-4)
    assert np.allclose(sim[1][2], 0.7071, atol=1e-4)
    assert np.allclose(sim[0][0], 1.0, atol=1e-4)

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
