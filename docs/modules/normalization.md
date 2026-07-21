# Normalization Module API Reference

The `nlpie` normalization module provides fast, Rust-backed methods for preparing embedding vectors before similarity computation, clustering, or visualization.

## `EmbeddingPreprocessor`

A convenience wrapper class that holds state (like epsilon value) and exposes all normalization routines.

```python
from nlpie import EmbeddingPreprocessor

preprocessor = EmbeddingPreprocessor(eps=1e-12)

# L2 normalize rows
normalized_matrix = preprocessor.l2_normalize_rows(embeddings)

# Mean center columns
centered_matrix, stats = preprocessor.mean_center(embeddings)

# Standardize columns (zero mean, unit variance)
standard_matrix, stats = preprocessor.standardize_columns(embeddings)

# Apply PCA whitening
whitened_matrix, model = preprocessor.whiten_pca(embeddings, n_components=128)

# Remove top principal components (debiasing)
debiased_matrix = preprocessor.remove_top_principal_components(embeddings, n_components=1)
```

## Functional API

All methods are also available as pure functions. They will return any fitted parameters (like mean vectors or PCA models) alongside the transformed data.

### `l2_normalize_rows(embeddings: MatrixLike, eps: float = 1e-12) -> list[list[float]]`
Divides each row by its L2 norm, making all vectors unit length. Essential before computing cosine similarity using dot products.

### `mean_center(embeddings: MatrixLike) -> Tuple[list[list[float]], FitStats]`
Subtracts the column-wise mean from each column. Returns the centered matrix and the `FitStats` (containing the computed mean).

### `standardize_columns(embeddings: MatrixLike, eps: float = 1e-12) -> Tuple[list[list[float]], FitStats]`
Centers the data and scales each column to have unit variance. Returns the standardized matrix and `FitStats` (mean and standard deviation). Raises `PreprocessingError` if a column has zero variance.

### `whiten_pca(embeddings: MatrixLike, n_components: Optional[int] = None, eps: float = 1e-12) -> Tuple[list[list[float]], WhitenModel]`
Applies PCA whitening, transforming the data such that its covariance matrix becomes the identity matrix. Useful for removing correlations between dimensions. Returns the whitened matrix and the `WhitenModel` (containing mean, projection matrix, and eigenvalues).

### `remove_top_principal_components(embeddings: MatrixLike, n_components: int) -> list[list[float]]`
Removes the specified number of top principal components from the embeddings. This is often used for debiasing embeddings by removing the dominant directions of variation.

## Return Types

### `FitStats`
A dataclass holding scaling statistics.
- `mean`: list of floats
- `std`: optional list of floats

### `WhitenModel`
A dataclass holding the PCA state.
- `mean`: list of floats (the centroid)
- `projection`: list of list of floats (eigenvectors)
- `eigenvalues`: list of floats
- `eps`: float
