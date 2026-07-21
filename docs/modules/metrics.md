# Metrics Module API Reference

The `nlpie` metrics module provides highly optimized Rust implementations of standard evaluation metrics for embeddings, clustering, dimensionality reduction, and retrieval.

## Basic Metrics

```python
from nlpie import (
    cosine_similarity, cosine_similarity_matrix, pairwise_cosine_stats,
    pearson_correlation, spearman_correlation,
)
```

- **`cosine_similarity(lhs, rhs)`**: Computes cosine similarity between two 1D vectors.
- **`cosine_similarity_matrix(embeddings)`**: Computes the full NxN cosine similarity matrix for a 2D embedding matrix.
- **`pairwise_cosine_stats(embeddings)`**: Computes all pairwise cosines in a single Rust pass and returns `(triangle, stats)` — the compact upper-triangle values plus `(mean, std, min, max)` — without materialising the NxN matrix in Python.
- **`pearson_correlation(x, y)`**: Computes the linear correlation between two 1D arrays.
- **`spearman_correlation(x, y)`**: Computes the rank correlation between two 1D arrays, handling ties gracefully.

## Clustering Metrics

```python
from nlpie import (
    adjusted_rand_index, normalized_mutual_info, purity_score, 
    calinski_harabasz_score, silhouette_score
)
```

**Label Comparison (Requires True and Predicted Labels):**
- **`adjusted_rand_index(labels_true, labels_pred)`**: Computes ARI. Perfect matching scores 1.0, random assignments score ~0.0.
- **`normalized_mutual_info(labels_true, labels_pred)`**: Computes NMI. Ranges from 0.0 (no mutual info) to 1.0 (perfect correlation).
- **`purity_score(labels_true, labels_pred)`**: Computes the purity of the clusters. 

**Intrinsic Quality (Requires Embeddings and Labels):**
- **`calinski_harabasz_score(embeddings, labels)`**: Variance Ratio Criterion. Higher values indicate better defined clusters.
- **`silhouette_score(embeddings, labels)`**: Mean Silhouette Coefficient. Ranges from -1.0 to 1.0. Near +1 indicates samples are far from neighboring clusters.

## Geometry and Pathology Diagnostics

```python
from nlpie import effective_rank, similarity_to_global_mean, compute_hubness
```

- **`effective_rank(embeddings)`**: Computes the entropy-based effective rank of the covariance matrix. Measures the true dimensionality utilized by the space.
- **`similarity_to_global_mean(embeddings)`**: Returns a list of cosine similarities of each point to the global centroid. Useful for detecting anisotropy.
- **`compute_hubness(embeddings, k=5)`**: Returns `(counts, skewness)`. Counts are the number of times each point appears in the K-NN sets of other points. Skewness is the hubness index (higher means more severe hubness pathology).

## Projection Quality Metrics

```python
from nlpie import trustworthiness, continuity, projection_quality
```

Evaluate dimensionality reduction (e.g., PCA, t-SNE, UMAP).

- **`trustworthiness(high_dim, low_dim, k=10)`**: Measures whether neighbors in the projected space were also neighbors in the original space. Penalizes false positives in the projection.
- **`continuity(high_dim, low_dim, k=10)`**: Measures whether neighbors in the original space are preserved in the projected space. Penalizes false negatives in the projection.
- **`projection_quality(high_dim, low_dim, k_values)`**: One-pass evaluation of both metrics for several neighbourhood sizes at once; returns a list of `(k, trustworthiness, continuity)` tuples. Both k-NN rank matrices are built once and reused, so this is much faster than calling the single-k functions in a loop.

> **Note:** `k` must satisfy `1 <= k < n` and `2n - 3k - 1 > 0` for `n`
> samples; out-of-range values raise `PreprocessingError` instead of being
> silently clamped.

## Retrieval and Ranking Metrics

```python
from nlpie import recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k, coverage_at_k
```

These methods operate on ranked lists of integer IDs.

- **`recall_at_k(retrieved, relevant, k)`**: `|relevant ∩ retrieved@K| / |relevant|`
- **`precision_at_k(retrieved, relevant, k)`**: `|relevant ∩ retrieved@K| / K`
- **`mean_reciprocal_rank(retrieved, relevant)`**: `1 / rank_of_first_relevant_item`
- **`ndcg_at_k(retrieved, relevant, k)`**: Normalized Discounted Cumulative Gain at K (binary relevance).
- **`coverage_at_k(all_retrieved, all_relevant, k)`**: Fraction of the total relevant item space that appears in at least one query's top-K list. Note that this accepts a *list of lists* (multiple queries).
