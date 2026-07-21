import math

import pytest
from nlpie import (
    EmbeddingPreprocessor,
    FitStats,
    PreprocessingError,
    WhitenModel,
    cosine_similarity,
    l2_normalize_rows,
    mean_center,
    remove_top_principal_components,
    standardize_columns,
    whiten_pca,
)


def test_cosine_similarity():
    # Orthogonal vectors
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-6)

    # Parallel vectors
    assert math.isclose(cosine_similarity([1.0, 2.0], [2.0, 4.0]), 1.0, abs_tol=1e-6)

    # Invalid length
    with pytest.raises(PreprocessingError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_l2_normalize_rows():
    embeddings = [[3.0, 4.0], [0.0, 0.0]]
    normalized = l2_normalize_rows(embeddings)
    assert math.isclose(normalized[0][0], 0.6, abs_tol=1e-6)
    assert math.isclose(normalized[0][1], 0.8, abs_tol=1e-6)
    assert normalized[1][0] == 0.0
    assert normalized[1][1] == 0.0


def test_mean_center():
    embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    centered, stats = mean_center(embeddings)

    assert isinstance(stats, FitStats)
    assert stats.mean == [3.0, 4.0]
    assert stats.std is None
    assert centered == [[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]


def test_standardize_columns():
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    standardized, stats = standardize_columns(embeddings)
    assert stats.mean == [2.0, 3.0]
    assert stats.std == [1.0, 1.0]
    assert standardized == [[-1.0, -1.0], [1.0, 1.0]]


def test_standardize_zero_std_dev():
    embeddings = [[1.0, 2.0], [1.0, 4.0]]
    with pytest.raises(PreprocessingError) as exc_info:
        standardize_columns(embeddings)
    assert "standard deviation is zero" in str(exc_info.value)


def test_whiten_pca():
    embeddings = [[1.0, 2.0], [3.0, 5.0], [5.0, 6.0]]
    whitened, model = whiten_pca(embeddings, n_components=2)
    assert isinstance(model, WhitenModel)
    assert len(whitened) == 3
    assert len(whitened[0]) == 2
    assert len(model.mean) == 2
    assert len(model.projection) == 2
    assert len(model.projection[0]) == 2
    assert len(model.eigenvalues) == 2


def test_remove_top_principal_components():
    embeddings = [[1.0, 2.0], [3.0, 5.0], [5.0, 6.0]]
    debiased = remove_top_principal_components(embeddings, n_components=1)
    assert len(debiased) == 3
    assert len(debiased[0]) == 2


def test_preprocessor_class():
    preprocessor = EmbeddingPreprocessor(eps=1e-4)
    embeddings = [[3.0, 4.0], [0.0, 0.0]]
    normalized = preprocessor.l2_normalize_rows(embeddings)
    assert math.isclose(normalized[0][0], 0.6, abs_tol=1e-6)

    # Empty matrix error
    with pytest.raises(PreprocessingError):
        preprocessor.mean_center([])

    # Mismatched row lengths error
    with pytest.raises(PreprocessingError):
        preprocessor.mean_center([[1.0, 2.0], [3.0]])
