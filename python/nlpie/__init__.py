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
    "NlpieError",
    "PreprocessingError",
]
