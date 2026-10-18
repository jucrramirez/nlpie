# Embedding Quality Report: Model B (Finetuned)

- **Samples:** 300
- **Dimensions:** 64
- **Status:** Critical

## Executive Summary

**Strong structure, weak retrieval.**

An embedding can look separated or well-conditioned while still placing task-relevant items too low in the nearest-neighbour ranking.

### Primary blockers

- Severe hubness detected (skewness=1.81). 43 point(s) appear in more than twice the expected neighbour lists.
- Very low effective rank (5.5 of 64 dimensions).
- Retrieval quality is very low (max Recall@K=0.050, max nDCG=0.026).

### Healthy signals

- Very low mean cosine similarity (-0.002) — points are nearly orthogonal.
- Clustering metrics are within acceptable ranges.
- Similarity to the global mean is 0.032.
- Projection quality is good (mean Trustworthiness=0.902, mean Continuity=0.915).

## Metric Interpretation

| Metric | Value | Status | Ideal direction | Reference |
|---|---:|---|---|---|
| intrinsic.mean | -0.0023 | Healthy | Lower alignment; values near 0 are typically desirable. | Warning > 0.50; critical > 0.80 |
| intrinsic.std | 0.4901 | Healthy | A sufficiently varied distribution; no universal target. | Warning < 0.05 |
| intrinsic.min | -0.5281 | Measured | Descriptive range; interpret with mean and standard deviation. | Descriptive only |
| intrinsic.max | 0.9744 | Measured | Descriptive range; interpret with mean and standard deviation. | Descriptive only |
| clustering.ari | N/A — not measured | Not measured | Higher is better; maximum 1.0. | Requires labels and predictions |
| clustering.nmi | N/A — not measured | Not measured | Higher is better; maximum 1.0. | Requires labels and predictions |
| clustering.purity | N/A — not measured | Not measured | Higher is better; maximum 1.0. | Requires labels and predictions |
| clustering.silhouette | 0.9425 | Healthy | Higher is better; maximum 1.0. | Warning < 0.25; critical < 0.10 |
| clustering.davies_bouldin | 0.0560 | Healthy | Lower is better; 0 is ideal. | Warning > 0.50; critical > 1.00 |
| geometry.effective_rank | 5.52 / 64 | Critical | Closer to ambient dimensions is better. | Warning < 38.4; critical < 19.2 |
| geometry.mean_similarity | 0.0317 | Healthy | Lower alignment with the global mean is generally better. | Warning > 0.50; critical > 0.80 |
| geometry.hubness_skewness | 1.8137 | Critical | Near 0 is better. | Warning > 0.50; critical > 1.00 |
| geometry.hubness_k | 5 | Measured | Context for the hubness calculation. | Configuration value |
| projection.trustworthiness@5 | 0.8969 | Healthy | Closer to 1.0 is better. | Warning < 0.80; critical < 0.50 |
| projection.continuity@5 | 0.9097 | Healthy | Closer to 1.0 is better. | Warning < 0.80; critical < 0.50 |
| projection.trustworthiness@10 | 0.9078 | Healthy | Closer to 1.0 is better. | Warning < 0.80; critical < 0.50 |
| projection.continuity@10 | 0.9194 | Healthy | Closer to 1.0 is better. | Warning < 0.80; critical < 0.50 |
| retrieval.recall@1 | 0.0% | Critical | Higher is better; maximum 100%. | Warning < 70%; critical < 30% |
| retrieval.precision@1 | 0.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.mrr@1 | 0.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.ndcg@1 | 0.0% | Critical | Higher is better; maximum 100%. | Warning < 70%; critical < 30% |
| retrieval.coverage@1 | 0.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.recall@3 | 0.0% | Critical | Higher is better; maximum 100%. | Warning < 70%; critical < 30% |
| retrieval.precision@3 | 0.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.mrr@3 | 0.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.ndcg@3 | 0.0% | Critical | Higher is better; maximum 100%. | Warning < 70%; critical < 30% |
| retrieval.coverage@3 | 0.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.recall@5 | 5.0% | Critical | Higher is better; maximum 100%. | Warning < 70%; critical < 30% |
| retrieval.precision@5 | 2.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.mrr@5 | 2.5% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |
| retrieval.ndcg@5 | 2.6% | Critical | Higher is better; maximum 100%. | Warning < 70%; critical < 30% |
| retrieval.coverage@5 | 5.0% | Measured | Higher is better; maximum 100%. | Higher is better; no dedicated severity threshold |

## Interpretation Details

- **hubness** (Critical): Severe hubness detected (skewness=1.81). 43 point(s) appear in more than twice the expected neighbour lists.
  - The K-NN distribution is heavily right-skewed — a small number of points (hubs) appear in the neighbour lists of a large fraction of other points. This can degrade retrieval quality and distort downstream evaluation.
  - Recommendation: Consider applying PCA whitening, contrastive loss post-processing, or removing hub points before downstream tasks.
- **effective_rank** (Critical): Very low effective rank (5.5 of 64 dimensions).
  - The embedding occupies a much lower intrinsic dimensionality (5.5) than its ambient dimension (64). This suggests severe redundancy — most dimensions are noise or highly correlated.
  - Recommendation: Consider PCA whitening or reducing embedding dimensionality. If using a decoder-based model, check for posterior collapse.
- **retrieval** (Critical): Retrieval quality is very low (max Recall@K=0.050, max nDCG=0.026).
  - The embedding space struggles to rank relevant items near the top.
  - Recommendation: Consider fine-tuning embeddings with a retrieval-aware loss (e.g., triplet loss, contrastive loss, or ListNet). Check if the embedding space is aligned with the retrieval task.
- **intrinsic** (Healthy): Very low mean cosine similarity (-0.002) — points are nearly orthogonal.
  - Near-orthogonal vectors indicate a high-capacity space with low redundancy.
  - Recommendation: No action required. The space appears well-conditioned.
- **clustering** (Healthy): Clustering metrics are within acceptable ranges.
  - ARI=N/A, NMI=N/A, Purity=N/A, Silhouette (cosine)=0.943, Davies-Bouldin (cosine)=0.056.
  - Recommendation: No action required.
- **similarity_to_mean** (Healthy): Similarity to the global mean is 0.032.
  - The space is well spread around the centroid.
  - Recommendation: No action required.
- **projection** (Healthy): Projection quality is good (mean Trustworthiness=0.902, mean Continuity=0.915).
  - The low-dimensional projection preserves neighbourhood structure well.
  - Recommendation: No action required.

## Full Metric Coverage

### Intrinsic Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| intrinsic.mean | -0.0023 | Average cosine similarity between pairs of vectors. |
| intrinsic.std | 0.4901 | Spread of pairwise similarities; a very narrow spread can indicate collapse. |
| intrinsic.min | -0.5281 | Lowest observed pairwise cosine similarity. |
| intrinsic.max | 0.9744 | Highest observed pairwise cosine similarity. |

### Clustering

| Metric | Value | Interpretation |
|---|---:|---|
| clustering.ari | N/A — not measured | ARI needs label data and is not available in this report. Not measured. |
| clustering.nmi | N/A — not measured | NMI needs label data and is not available in this report. Not measured. |
| clustering.purity | N/A — not measured | Purity needs label data and is not available in this report. Not measured. |
| clustering.silhouette | 0.9425 | Measures how separated and internally coherent clusters are. |
| clustering.davies_bouldin | 0.0560 | Measures cluster overlap; lower values mean cleaner separation. |

### Geometry & Pathology

| Metric | Value | Interpretation |
|---|---:|---|
| geometry.effective_rank | 5.52 / 64 | Estimates how many dimensions carry meaningful variation. |
| geometry.mean_similarity | 0.0317 | Measures whether vectors point toward one dominant global direction. |
| geometry.hubness_skewness | 1.8137 | Measures whether a small number of points appear in too many neighbour lists. |
| geometry.hubness_k | 5 | Neighbourhood size used to measure hubness. |

### Projection

| Metric | Value | Interpretation |
|---|---:|---|
| projection.trustworthiness@5 | 0.8969 | Measures whether nearby high-dimensional neighbours stay nearby in the projection. |
| projection.continuity@5 | 0.9097 | Measures whether projection neighbours reflect relationships in the original space. |
| projection.trustworthiness@10 | 0.9078 | Measures whether nearby high-dimensional neighbours stay nearby in the projection. |
| projection.continuity@10 | 0.9194 | Measures whether projection neighbours reflect relationships in the original space. |

### Retrieval

| Metric | Value | Interpretation |
|---|---:|---|
| retrieval.recall@1 | 0.0% | Share of relevant items retrieved in the top K. |
| retrieval.precision@1 | 0.0% | Share of the top K results that are relevant. |
| retrieval.mrr@1 | 0.0% | Rewards putting the first relevant result near the top. |
| retrieval.ndcg@1 | 0.0% | Rewards ranking relevant results early, with graded relevance support. |
| retrieval.coverage@1 | 0.0% | Share of queries for which at least one relevant item was found. |
| retrieval.recall@3 | 0.0% | Share of relevant items retrieved in the top K. |
| retrieval.precision@3 | 0.0% | Share of the top K results that are relevant. |
| retrieval.mrr@3 | 0.0% | Rewards putting the first relevant result near the top. |
| retrieval.ndcg@3 | 0.0% | Rewards ranking relevant results early, with graded relevance support. |
| retrieval.coverage@3 | 0.0% | Share of queries for which at least one relevant item was found. |
| retrieval.recall@5 | 5.0% | Share of relevant items retrieved in the top K. |
| retrieval.precision@5 | 2.0% | Share of the top K results that are relevant. |
| retrieval.mrr@5 | 2.5% | Rewards putting the first relevant result near the top. |
| retrieval.ndcg@5 | 2.6% | Rewards ranking relevant results early, with graded relevance support. |
| retrieval.coverage@5 | 5.0% | Share of queries for which at least one relevant item was found. |

## Recommended Next Steps

1. Consider applying PCA whitening, contrastive loss post-processing, or removing hub points before downstream tasks.
2. No action required. The space appears well-conditioned.
3. Consider PCA whitening or reducing embedding dimensionality. If using a decoder-based model, check for posterior collapse.
4. Consider fine-tuning embeddings with a retrieval-aware loss (e.g., triplet loss, contrastive loss, or ListNet). Check if the embedding space is aligned with the retrieval task.

## Evidence Availability

| Evidence | Available |
|---|---|
| intrinsic | Yes |
| clustering | Yes |
| geometry | Yes |
| projection | Yes |
| retrieval | Yes |
| pairwise_similarities | Yes |
| hubness_counts | Yes |
| sim_to_mean | Yes |

## How to Read This Report

- **Intrinsic geometry:** describes how similar vectors are to one another and whether the space is overly aligned.
- **Clustering:** describes separation and overlap between groups; Silhouette is higher-is-better and Davies–Bouldin is lower-is-better.
- **Geometry:** describes usable dimensionality and whether a few vectors dominate neighbourhoods.
- **Projection:** describes how faithfully a lower-dimensional view preserves neighbourhood relationships.
- **Retrieval:** describes whether relevant items are found and ranked near the top; higher values are better and 100% is the maximum.
- **Not measured:** means the required input or raw trace was not present; it is not a failing score.
