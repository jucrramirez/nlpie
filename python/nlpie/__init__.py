"""nlpie: High-performance NLP embedding preprocessing and normalization library."""

from ._api import (
    EmbeddingPreprocessor,
    FitStats,
    WhitenModel,
    # Clustering metrics
    adjusted_rand_index,
    calinski_harabasz_score,
    compute_hubness,
    continuity,
    cosine_similarity,
    # Basic metrics
    cosine_similarity_matrix,
    coverage_at_k,
    # Geometry and hubness metrics
    effective_rank,
    l2_normalize_rows,
    mean_center,
    mean_reciprocal_rank,
    ndcg_at_k,
    normalized_mutual_info,
    pairwise_cosine_stats,
    pearson_correlation,
    precision_at_k,
    projection_quality,
    purity_score,
    # Retrieval and ranking metrics (TASK-006)
    recall_at_k,
    remove_top_principal_components,
    silhouette_score,
    similarity_to_global_mean,
    spearman_correlation,
    standardize_columns,
    # Projection quality metrics (TASK-005)
    trustworthiness,
    whiten_pca,
)
from ._dashboard import (
    plot_comparison_delta,
    plot_comparison_grouped_bar,
    plot_comparison_radar,
    plot_correlation_heatmap,
    plot_eigenvalue_scree,
    plot_embedding_scatter,
    plot_hubness_bar,
    plot_hubness_histogram,
    plot_projection_quality,
    plot_quality_report,
    plot_retrieval_metrics,
    plot_silhouette,
    plot_similarity_distribution,
    plot_similarity_heatmap,
    plot_similarity_to_mean,
)
from ._errors import NlpieError, PreprocessingError
from .backends import Dashboard, PlotBackend, PlotlyBackend
from .dashboard.builder import DashboardBuilder
from .dashboard.comparison import (
    compare_and_plot_delta,
    compare_and_plot_grouped_bar,
    compare_and_plot_radar,
)
from .export import HtmlExporter, JsonExporter, MarkdownExporter, ReportExporter
from .interpret import (
    Explanation,
    ExplanationProvider,
    ExplanationRegistry,
    HubInfo,
    HubnessExplanation,
    InterpretationReport,
    explain_hubness,
)
from .metrics import (
    ClusteringMetrics,
    EmbeddingQualityReport,
    GeometryMetrics,
    IntrinsicMetrics,
    ProjectionMetrics,
    RetrievalMetrics,
    compare_models,
    evaluate_embedding_quality,
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
    "pairwise_cosine_stats",
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
    "projection_quality",
    # Retrieval and ranking metrics (TASK-006)
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "coverage_at_k",
    # Quality reports
    "EmbeddingQualityReport",
    "IntrinsicMetrics",
    "ClusteringMetrics",
    "GeometryMetrics",
    "ProjectionMetrics",
    "RetrievalMetrics",
    "evaluate_embedding_quality",
    "compare_models",
    # Interpret
    "Explanation",
    "ExplanationProvider",
    "ExplanationRegistry",
    "HubnessExplanation",
    "HubInfo",
    "InterpretationReport",
    "explain_hubness",
    # Backends
    "Dashboard",
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
    # Export
    "ReportExporter",
    "HtmlExporter",
    "JsonExporter",
    "MarkdownExporter",
    "NlpieError",
    "PreprocessingError",
]
