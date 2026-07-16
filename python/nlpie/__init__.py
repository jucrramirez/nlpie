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
    cosine_similarity_matrix_stats,
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
from .config import NlpieConfig
from .interpret import Explanation, ExplanationProvider, ExplanationRegistry, HubnessExplanation, HubInfo, InterpretationReport, explain_hubness
from .backends import PlotBackend, PlotlyBackend
from .export import ReportExporter, HtmlExporter, JsonExporter, MarkdownExporter
from .dashboard.builder import DashboardBuilder
from .dashboard.comparison import (
    compare_and_plot_radar,
    compare_and_plot_grouped_bar,
    compare_and_plot_delta,
)
from ._dashboard import (
    plot_hubness_histogram,
    plot_similarity_distribution,
    plot_similarity_to_mean,
    plot_projection_quality,
    plot_retrieval_metrics,
    plot_similarity_heatmap,
    plot_hubness_bar,
    plot_eigenvalue_scree,
    plot_silhouette,
    plot_correlation_heatmap,
    plot_embedding_scatter,
    plot_comparison_radar,
    plot_comparison_grouped_bar,
    plot_comparison_delta,
    plot_quality_report,
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
    "cosine_similarity_matrix_stats",
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
    "Explanation",
    "ExplanationProvider",
    "ExplanationRegistry",
    "HubnessExplanation",
    "HubInfo",
    "InterpretationReport",
    "explain_hubness",
    # Backends
    "PlotBackend",
    "PlotlyBackend",
    # Dashboard
    "DashboardBuilder",
    "compare_and_plot_radar",
    "compare_and_plot_grouped_bar",
    "compare_and_plot_delta",
    "plot_hubness_histogram",
    "plot_similarity_distribution",
    "plot_similarity_to_mean",
    "plot_projection_quality",
    "plot_retrieval_metrics",
    "plot_similarity_heatmap",
    "plot_hubness_bar",
    "plot_eigenvalue_scree",
    "plot_silhouette",
    "plot_correlation_heatmap",
    "plot_embedding_scatter",
    "plot_comparison_radar",
    "plot_comparison_grouped_bar",
    "plot_comparison_delta",
    "plot_quality_report",
    # Config
    "NlpieConfig",
    # Export
    "ReportExporter",
    "HtmlExporter",
    "JsonExporter",
    "MarkdownExporter",
    "NlpieError",
    "PreprocessingError",
]
