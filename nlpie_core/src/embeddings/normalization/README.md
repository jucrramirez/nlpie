# Embeddings Normalization Module

This Rust module provides high-performance routines for processing and normalizing embedding matrices, leveraging `ndarray` and `ndarray-linalg` (LAPACK).

## Features

- **L2 Normalization (`l2_normalize_rows`)**: Scales each embedding vector to have a unit length (L2 norm of 1). This is crucial for cosine similarity calculations.
- **Mean Centering (`mean_center`)**: Subtracts the column-wise mean from the embeddings, centering the dataset around the origin.
- **Standardization (`standardize_columns`)**: Centers the dataset and scales each column to have unit variance (Z-score normalization).
- **PCA Whitening (`whiten_pca`)**: Decorrelates the embedding features and scales them to have unit variance across principal components. It projects the data onto its eigenvectors and normalizes the eigenvalues.
- **Top Principal Component Removal (`remove_top_principal_components`)**: Helps debias embeddings by removing the top `k` principal directions of variation (often associated with frequency/stopword biases in NLP).
- **Cosine Similarity (`cosine_similarity`)**: High-performance cosine similarity computation between two slice vectors.

## Architecture

The module is broken into several submodules for maintainability:
- `error.rs`: Defines `PreprocessingError` for standard error handling.
- `utils.rs`: Small helper functions and metrics like `cosine_similarity` and `l2_normalize_rows`.
- `centering.rs`: Contains the `mean_center` implementation and `FitStats`.
- `standardization.rs`: Contains the `standardize_columns` implementation.
- `whitening.rs`: Contains PCA-related functionality (`whiten_pca`, `remove_top_principal_components`, and `WhitenModel`).
- `preprocessor.rs`: Exposes `EmbeddingPreprocessor` which acts as a convenient wrapper around these functions, storing configuration state like `eps`.

## Usage

```rust
use ndarray::Array2;
use nlpie_core::embeddings::normalization::EmbeddingPreprocessor;

// Initialize the preprocessor with a default epsilon (1e-12)
let preprocessor = EmbeddingPreprocessor::default();

// Example matrix (samples, features)
// let embeddings: Array2<f32> = ...;

// L2 Normalize
// let normalized = preprocessor.l2_normalize_rows(&embeddings);

// Whiten PCA
// let (whitened, model) = preprocessor.whiten_pca(&embeddings, Some(128)).unwrap();
```

## Best Practices

- Make sure to compile with an appropriate BLAS/LAPACK backend (e.g., `openblas-system`) for optimal performance when computing PCA.
- When performing sequential normalization steps, ensure the semantic meaning of your embeddings is preserved. For instance, L2 normalization is usually the final step before calculating cosine similarities.
