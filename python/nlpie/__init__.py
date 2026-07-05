"""nlpie: High-performance NLP embedding preprocessing and normalization library."""

from ._api import (
    EmbeddingPreprocessor,
    FitStats,
    WhitenModel,
    cosine_similarity,
    l2_normalize_rows,
    mean_center,
    standardize_columns,
    whiten_pca,
    remove_top_principal_components,
    # Projection quality metrics (TASK-005)
    trustworthiness,
    continuity,
    # Retrieval and ranking metrics (TASK-006)
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    coverage_at_k,
)
from ._errors import NlpieError, PreprocessingError

__all__ = [
    "EmbeddingPreprocessor",
    "FitStats",
    "WhitenModel",
    "cosine_similarity",
    "l2_normalize_rows",
    "mean_center",
    "standardize_columns",
    "whiten_pca",
    "remove_top_principal_components",
    # Projection quality metrics (TASK-005)
    "trustworthiness",
    "continuity",
    # Retrieval and ranking metrics (TASK-006)
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "coverage_at_k",
    "NlpieError",
    "PreprocessingError",
]
