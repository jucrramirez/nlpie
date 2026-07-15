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
    # Basic metrics
    cosine_similarity_matrix,
    pearson_correlation,
    spearman_correlation,
    # Clustering metrics
    adjusted_rand_index,
    normalized_mutual_info,
    purity_score,
    calinski_harabasz_score,
    silhouette_score,
    # Geometry and hubness metrics
    effective_rank,
    similarity_to_global_mean,
    compute_hubness,
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
from .interpret import HubnessExplanation, HubInfo, explain_hubness
from .backends import PlotBackend, PlotlyBackend
from .dashboard import (
    plot_hubness_histogram,
    plot_similarity_distribution,
    plot_similarity_to_mean,
    plot_projection_quality,
    plot_retrieval_metrics,
)

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
    # Basic metrics
    "cosine_similarity_matrix",
    "pearson_correlation",
    "spearman_correlation",
    # Clustering metrics
    "adjusted_rand_index",
    "normalized_mutual_info",
    "purity_score",
    "calinski_harabasz_score",
    "silhouette_score",
    # Geometry and hubness metrics
    "effective_rank",
    "similarity_to_global_mean",
    "compute_hubness",
    # Projection quality metrics (TASK-005)
    "trustworthiness",
    "continuity",
    # Retrieval and ranking metrics (TASK-006)
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "coverage_at_k",
    # Interpret
    "HubnessExplanation",
    "HubInfo",
    "explain_hubness",
    # Backends
    "PlotBackend",
    "PlotlyBackend",
    # Dashboard
    "plot_hubness_histogram",
    "plot_similarity_distribution",
    "plot_similarity_to_mean",
    "plot_projection_quality",
    "plot_retrieval_metrics",
    "NlpieError",
    "PreprocessingError",
]
