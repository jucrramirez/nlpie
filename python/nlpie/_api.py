"""Public high-level Python API for the nlpie embedding normalization library."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps

import numpy as np

# Try importing the compiled binary extension module
try:
    from . import _nlpie_core
except ImportError as e:
    raise ImportError(
        "Failed to import the compiled native extension '_nlpie_core'. "
        "Please make sure the library was compiled and installed correctly. "
        "Hint: run 'uv pip install -e .' to build it locally."
    ) from e

from ._errors import PreprocessingError
from ._types import MatrixLike, VectorLike


@dataclass(frozen=True)
class FitStats:
    """Statistics calculated during embedding normalization operations.

    Attributes:
        mean: The column-wise mean vector of the dataset.
        std: The column-wise standard deviation vector (if computed).
    """

    mean: list[float]
    std: list[float] | None = None

    @classmethod
    def _from_rust(cls, rust_stats) -> "FitStats":
        return cls(mean=rust_stats.mean, std=rust_stats.std)


@dataclass(frozen=True)
class WhitenModel:
    """Model state holding PCA projection parameters.

    Attributes:
        mean: The mean vector used to center the data before projection.
        projection: The projection matrix (eigenvectors) used to transform the data.
        eigenvalues: The eigenvalues corresponding to the principal components.
        eps: The epsilon value used to avoid division by zero.
    """

    mean: list[float]
    projection: list[list[float]]
    eigenvalues: list[float]
    eps: float

    @classmethod
    def _from_rust(cls, rust_model) -> "WhitenModel":
        return cls(
            mean=rust_model.mean,
            projection=rust_model.projection,
            eigenvalues=rust_model.eigenvalues,
            eps=rust_model.eps,
        )


def _to_matrix(embeddings: MatrixLike) -> np.ndarray:
    """Converts a MatrixLike input to a float32 numpy array (zero-copy if already numpy).

    Supports lists, tuples, and numpy arrays. Empty input is treated as (0, 0).
    """
    if isinstance(embeddings, (list, tuple)) and len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise TypeError(f"Embeddings must be a 2D matrix, got {arr.ndim}D array")
    return arr


def _to_vector(vector: VectorLike) -> list[float]:
    """Converts a VectorLike input to a standard list of floats."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    if not isinstance(vector, (list, tuple)):
        raise TypeError("Vector must be a list, tuple, or numpy array")

    return [float(x) for x in vector]


def _wrap_exceptions(func):
    """Decorator to intercept PyO3 value/overflow errors and raise PreprocessingError."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise PreprocessingError(str(e)) from e
        except OverflowError as e:
            # Raised by PyO3 when e.g. a negative label/ID does not fit the
            # Rust parameter type (usize) — surface it as a domain error.
            raise PreprocessingError(f"Value out of range for the native extension: {e}") from e

    return wrapper


@_wrap_exceptions
def cosine_similarity(lhs: VectorLike, rhs: VectorLike) -> float:
    """Computes the cosine similarity between two vectors.

    Args:
        lhs: The first vector.
        rhs: The second vector.

    Returns:
        The cosine similarity float value.

    Raises:
        PreprocessingError: If input shapes are invalid or vector length is zero.
    """
    return _nlpie_core.cosine_similarity(_to_vector(lhs), _to_vector(rhs))


@_wrap_exceptions
def cosine_similarity_matrix(embeddings: MatrixLike, eps: float = 1e-12) -> list[list[float]]:
    """Computes the cosine similarity matrix for a set of embeddings.

    Args:
        embeddings: The 2D embedding matrix.
        eps: Small value to prevent division by zero during normalization.

    Returns:
        An N x N cosine similarity matrix.
    """
    return _nlpie_core.cosine_similarity_matrix(_to_matrix(embeddings), eps)


@_wrap_exceptions
def pairwise_cosine_stats(
    embeddings: MatrixLike, eps: float = 1e-12
) -> tuple[list[float], float, float, float, float]:
    """Computes all pairwise cosine similarities and their summary statistics.

    The flattened upper triangle (i < j, row-major order, exactly
    ``n * (n - 1) / 2`` values) plus mean, std, min, and max are computed in a
    single Rust pass — the full N x N matrix never crosses into Python. Use
    :func:`cosine_similarity_matrix` when the full matrix is actually needed.

    Args:
        embeddings: The 2D embedding matrix.
        eps: Small value to prevent division by zero during normalization.

    Returns:
        A tuple (upper_triangle, mean, std, min, max).
    """
    return _nlpie_core.pairwise_cosine_stats(_to_matrix(embeddings), eps)


@_wrap_exceptions
def pearson_correlation(x: VectorLike, y: VectorLike) -> float:
    """Computes the Pearson correlation coefficient between two vectors.

    Args:
        x: The first vector.
        y: The second vector.

    Returns:
        The Pearson correlation coefficient.
    """
    return _nlpie_core.pearson_correlation(_to_vector(x), _to_vector(y))


@_wrap_exceptions
def spearman_correlation(x: VectorLike, y: VectorLike) -> float:
    """Computes the Spearman rank correlation coefficient between two vectors.

    Args:
        x: The first vector.
        y: The second vector.

    Returns:
        The Spearman rank correlation coefficient.
    """
    return _nlpie_core.spearman_correlation(_to_vector(x), _to_vector(y))


@_wrap_exceptions
def l2_normalize_rows(embeddings: MatrixLike, eps: float = 1e-12) -> list[list[float]]:
    """Applies L2 normalization to each row of the embedding matrix.

    Args:
        embeddings: The 2D embedding matrix.
        eps: Small value to prevent division by zero.

    Returns:
        The L2-normalized 2D embedding matrix as nested list.
    """
    return _nlpie_core.l2_normalize_rows(_to_matrix(embeddings), eps)


@_wrap_exceptions
def mean_center(embeddings: MatrixLike) -> tuple[list[list[float]], FitStats]:
    """Centers the embedding matrix by subtracting the mean of each column.

    Args:
        embeddings: The 2D embedding matrix.

    Returns:
        A tuple of (centered_matrix, FitStats).

    Raises:
        PreprocessingError: If the input matrix is empty or invalid.
    """
    centered, rust_stats = _nlpie_core.mean_center(_to_matrix(embeddings))
    return centered, FitStats._from_rust(rust_stats)


@_wrap_exceptions
def standardize_columns(
    embeddings: MatrixLike, eps: float = 1e-12
) -> tuple[list[list[float]], FitStats]:
    """Standardizes columns of the embedding matrix (zero mean, unit variance).

    Args:
        embeddings: The 2D embedding matrix.
        eps: Small value to prevent division by zero in case of zero variance.

    Returns:
        A tuple of (standardized_matrix, FitStats).

    Raises:
        PreprocessingError: If a column has standard deviation below eps or if the matrix is empty.
    """
    standardized, rust_stats = _nlpie_core.standardize_columns(_to_matrix(embeddings), eps)
    return standardized, FitStats._from_rust(rust_stats)


@_wrap_exceptions
def whiten_pca(
    embeddings: MatrixLike, n_components: int | None = None, eps: float = 1e-12
) -> tuple[list[list[float]], WhitenModel]:
    """Applies PCA whitening transformation to the embeddings.

    Args:
        embeddings: The 2D embedding matrix.
        n_components: The number of principal components to keep (defaults to min(nrows, ncols)).
        eps: Small value to prevent division by zero during scaling.

    Returns:
        A tuple of (whitened_matrix, WhitenModel).

    Raises:
        PreprocessingError: If calculation fails or component count is invalid.
    """
    whitened, rust_model = _nlpie_core.whiten_pca(_to_matrix(embeddings), n_components, eps)
    return whitened, WhitenModel._from_rust(rust_model)


@_wrap_exceptions
def remove_top_principal_components(embeddings: MatrixLike, n_components: int) -> list[list[float]]:
    """Removes the top principal components from the embedding matrix (debiasing).

    Args:
        embeddings: The 2D embedding matrix.
        n_components: The number of top components to remove.

    Returns:
        The debiased 2D embedding matrix.

    Raises:
        PreprocessingError: If calculations fail or component count is invalid.
    """
    return _nlpie_core.remove_top_principal_components(_to_matrix(embeddings), n_components)


class EmbeddingPreprocessor:
    """Premium convenience preprocessor wrapping all embedding normalization routines.

    Holds state such as the standard epsilon value used across methods.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Initializes the preprocessor.

        Args:
            eps: Epsilon value to prevent division by zero in standard dev or PCA whitening.
        """
        self._preprocessor = _nlpie_core.EmbeddingPreprocessor(eps)
        self.eps = eps

    @_wrap_exceptions
    def l2_normalize_rows(self, embeddings: MatrixLike) -> list[list[float]]:
        """Applies L2 normalization to each row of the embedding matrix.

        Args:
            embeddings: The 2D embedding matrix.

        Returns:
            The normalized 2D embedding matrix.
        """
        return self._preprocessor.l2_normalize_rows(_to_matrix(embeddings))

    @_wrap_exceptions
    def mean_center(self, embeddings: MatrixLike) -> tuple[list[list[float]], FitStats]:
        """Centers the embedding matrix by subtracting the mean of each column.

        Args:
            embeddings: The 2D embedding matrix.

        Returns:
            A tuple of (centered_matrix, FitStats).
        """
        centered, rust_stats = self._preprocessor.mean_center(_to_matrix(embeddings))
        return centered, FitStats._from_rust(rust_stats)

    @_wrap_exceptions
    def standardize_columns(self, embeddings: MatrixLike) -> tuple[list[list[float]], FitStats]:
        """Standardizes columns of the embedding matrix to zero mean and unit variance.

        Args:
            embeddings: The 2D embedding matrix.

        Returns:
            A tuple of (standardized_matrix, FitStats).
        """
        standardized, rust_stats = self._preprocessor.standardize_columns(_to_matrix(embeddings))
        return standardized, FitStats._from_rust(rust_stats)

    @_wrap_exceptions
    def whiten_pca(
        self, embeddings: MatrixLike, n_components: int | None = None
    ) -> tuple[list[list[float]], WhitenModel]:
        """Applies PCA whitening to the embeddings.

        Args:
            embeddings: The 2D embedding matrix.
            n_components: Number of components to keep (default is all available).

        Returns:
            A tuple of (whitened_matrix, WhitenModel).
        """
        whitened, rust_model = self._preprocessor.whiten_pca(_to_matrix(embeddings), n_components)
        return whitened, WhitenModel._from_rust(rust_model)

    @_wrap_exceptions
    def remove_top_principal_components(
        self, embeddings: MatrixLike, n_components: int
    ) -> list[list[float]]:
        """Removes the top principal components from the embeddings (debiasing).

        Args:
            embeddings: The 2D embedding matrix.
            n_components: Number of top components to remove.

        Returns:
            The debiased 2D embedding matrix.
        """
        return self._preprocessor.remove_top_principal_components(
            _to_matrix(embeddings), n_components
        )


# =============================================================================
# Clustering Metrics
# =============================================================================


@_wrap_exceptions
def adjusted_rand_index(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Computes the Adjusted Rand Index (ARI)."""
    return _nlpie_core.adjusted_rand_index(list(labels_true), list(labels_pred))


@_wrap_exceptions
def normalized_mutual_info(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Computes the Normalized Mutual Information (NMI)."""
    return _nlpie_core.normalized_mutual_info(list(labels_true), list(labels_pred))


@_wrap_exceptions
def purity_score(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Computes the Purity score."""
    return _nlpie_core.purity_score(list(labels_true), list(labels_pred))


@_wrap_exceptions
def calinski_harabasz_score(embeddings: MatrixLike, labels: Sequence[int]) -> float:
    """Computes the Calinski-Harabasz index."""
    return _nlpie_core.calinski_harabasz_score(_to_matrix(embeddings), list(labels))


@_wrap_exceptions
def silhouette_score(embeddings: MatrixLike, labels: Sequence[int]) -> float:
    """Computes the mean Silhouette Coefficient."""
    return _nlpie_core.silhouette_score(_to_matrix(embeddings), list(labels))


# =============================================================================
# Geometry & Pathology Metrics
# =============================================================================


@_wrap_exceptions
def effective_rank(embeddings: MatrixLike) -> float:
    """Computes the effective rank of an embedding space."""
    return _nlpie_core.effective_rank(_to_matrix(embeddings))


@_wrap_exceptions
def similarity_to_global_mean(embeddings: MatrixLike) -> list[float]:
    """Computes the cosine similarity of each point to the global centroid."""
    return _nlpie_core.similarity_to_global_mean(_to_matrix(embeddings))


@_wrap_exceptions
def compute_hubness(embeddings: MatrixLike, k: int = 5) -> tuple[list[int], float]:
    """Computes exact K-Nearest Neighbors hubness counts and skewness.

    Returns:
        A tuple of (hubness_counts, skewness).
    """
    return _nlpie_core.compute_hubness(_to_matrix(embeddings), k)


# =============================================================================
# Projection Quality Metrics  (TASK-005)
# =============================================================================


@_wrap_exceptions
def trustworthiness(
    high_dim: MatrixLike,
    low_dim: MatrixLike,
    k: int = 10,
) -> float:
    """Computes the trustworthiness of a low-dimensional projection.

    Measures whether the K nearest neighbours in the projection space were also
    close in the original high-dimensional space. A score of 1.0 is perfect.

    Args:
        high_dim: Embedding matrix in the original space ``(n_samples, d_high)``.
        low_dim:  Embedding matrix in the projected space ``(n_samples, d_low)``.
        k:        Number of neighbours to consider (default ``10``).

    Returns:
        A float in ``[0, 1]``, where ``1.0`` means a perfect projection.

    Raises:
        PreprocessingError: If shapes are inconsistent or ``k`` is invalid.
    """
    return _nlpie_core.trustworthiness(_to_matrix(high_dim), _to_matrix(low_dim), k)


@_wrap_exceptions
def continuity(
    high_dim: MatrixLike,
    low_dim: MatrixLike,
    k: int = 10,
) -> float:
    """Computes the continuity of a low-dimensional projection.

    Measures whether the K nearest neighbours in the original space are preserved
    in the projected space. A score of 1.0 is perfect.

    Args:
        high_dim: Embedding matrix in the original space ``(n_samples, d_high)``.
        low_dim:  Embedding matrix in the projected space ``(n_samples, d_low)``.
        k:        Number of neighbours to consider (default ``10``).

    Returns:
        A float in ``[0, 1]``, where ``1.0`` means a perfect projection.

    Raises:
        PreprocessingError: If shapes are inconsistent or ``k`` is invalid.
    """
    return _nlpie_core.continuity(_to_matrix(high_dim), _to_matrix(low_dim), k)


@_wrap_exceptions
def projection_quality(
    high_dim: MatrixLike,
    low_dim: MatrixLike,
    k_values: Sequence[int],
) -> list[tuple[int, float, float]]:
    """Computes trustworthiness and continuity for several ``k`` values in one pass.

    Both O(N²) k-NN rank matrices are built exactly once in Rust and reused
    for every ``k``, instead of being rebuilt per metric per ``k``.

    Args:
        high_dim: Embedding matrix in the original space ``(n_samples, d_high)``.
        low_dim:  Embedding matrix in the projected space ``(n_samples, d_low)``.
        k_values: Neighbourhood sizes to evaluate.

    Returns:
        A list of ``(k, trustworthiness, continuity)`` triples in the same
        order as ``k_values``.

    Raises:
        PreprocessingError: If shapes are inconsistent, ``k_values`` is empty,
            or any ``k`` is outside the valid domain.
    """
    return _nlpie_core.projection_quality(_to_matrix(high_dim), _to_matrix(low_dim), list(k_values))


# =============================================================================
# Retrieval and Ranking Metrics  (TASK-006)
# =============================================================================


@_wrap_exceptions
def recall_at_k(
    retrieved: Sequence[int],
    relevant: Sequence[int],
    k: int,
) -> float:
    """Computes Recall\\@K for a single query.

    Recall\\@K = |relevant ∩ retrieved\\@K| / |relevant|.

    Args:
        retrieved: Ranked list of document IDs, most-relevant first.
        relevant:  Ground-truth relevant document IDs.
        k:         Cut-off rank.

    Returns:
        Recall score in ``[0, 1]``.

    Raises:
        PreprocessingError: If ``k`` is zero or ``retrieved`` is empty.
    """
    return _nlpie_core.recall_at_k(list(retrieved), list(relevant), k)


@_wrap_exceptions
def precision_at_k(
    retrieved: Sequence[int],
    relevant: Sequence[int],
    k: int,
) -> float:
    """Computes Precision\\@K for a single query.

    Precision\\@K = |relevant ∩ retrieved\\@K| / K.

    Args:
        retrieved: Ranked list of document IDs, most-relevant first.
        relevant:  Ground-truth relevant document IDs.
        k:         Cut-off rank.

    Returns:
        Precision score in ``[0, 1]``.

    Raises:
        PreprocessingError: If ``k`` is zero or ``retrieved`` is empty.
    """
    return _nlpie_core.precision_at_k(list(retrieved), list(relevant), k)


@_wrap_exceptions
def mean_reciprocal_rank(
    retrieved: Sequence[int],
    relevant: Sequence[int],
) -> float:
    """Computes Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank_of_first_relevant_item, or ``0.0`` if no relevant item appears.

    Args:
        retrieved: Ranked list of document IDs, most-relevant first.
        relevant:  Ground-truth relevant document IDs.

    Returns:
        MRR score in ``[0, 1]``.

    Raises:
        PreprocessingError: If ``retrieved`` is empty.
    """
    return _nlpie_core.mean_reciprocal_rank(list(retrieved), list(relevant))


@_wrap_exceptions
def ndcg_at_k(
    retrieved: Sequence[int],
    relevant: Sequence[int],
    k: int,
) -> float:
    """Computes normalised Discounted Cumulative Gain (nDCG\\@K) for a single query.

    Uses binary relevance judgements (1 if relevant, 0 otherwise).

    Args:
        retrieved: Ranked list of document IDs, most-relevant first.
        relevant:  Ground-truth relevant document IDs.
        k:         Cut-off rank.

    Returns:
        nDCG score in ``[0, 1]``.

    Raises:
        PreprocessingError: If ``k`` is zero or ``retrieved`` is empty.
    """
    return _nlpie_core.ndcg_at_k(list(retrieved), list(relevant), k)


@_wrap_exceptions
def coverage_at_k(
    all_retrieved: Sequence[Sequence[int]],
    all_relevant: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Computes Coverage\\@K across multiple queries.

    Coverage\\@K = fraction of the total relevant-item space that appears in
    at least one query's top-K retrieved list.

    Args:
        all_retrieved: One ranked list per query.
        all_relevant:  One relevant-ID list per query.
        k:             Cut-off rank.

    Returns:
        Coverage score in ``[0, 1]``.

    Raises:
        PreprocessingError: If ``k`` is zero, lists are mismatched, or empty.
    """
    return _nlpie_core.coverage_at_k(
        [list(r) for r in all_retrieved],
        [list(r) for r in all_relevant],
        k,
    )
