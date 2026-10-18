"""nlpie: High-performance NLP embedding preprocessing and normalization library."""

from importlib import import_module
from typing import TYPE_CHECKING

from ._api import (
    EmbeddingPreprocessor,
    FitStats,
    WhitenModel,
    # Clustering metrics
    adjusted_rand_index,
    compute_hubness,
    continuity,
    cosine_similarity,
    # Basic metrics
    cosine_similarity_matrix,
    coverage_at_k,
    davies_bouldin_score,
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
from ._errors import NlpieError, PreprocessingError

if TYPE_CHECKING:
    from ._dashboard import (  # noqa: F401
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
    from .backends import Dashboard, PlotBackend, PlotlyBackend  # noqa: F401
    from .dashboard.builder import DashboardBuilder  # noqa: F401
    from .dashboard.comparison import (  # noqa: F401
        compare_and_plot_delta,
        compare_and_plot_grouped_bar,
        compare_and_plot_radar,
    )
    from .export import (  # noqa: F401
        HtmlExporter,
        JsonExporter,
        MarkdownExporter,
        ReportExporter,
    )
    from .interpret import (  # noqa: F401
        Explanation,
        ExplanationProvider,
        ExplanationRegistry,
        HubInfo,
        HubnessExplanation,
        InterpretationReport,
        explain_hubness,
    )
    from .metrics import (  # noqa: F401
        ClusteringMetrics,
        EmbeddingQualityReport,
        GeometryMetrics,
        IntrinsicMetrics,
        ProjectionMetrics,
        RetrievalMetrics,
        compare_models,
        evaluate_embedding_quality,
    )

# Keep the public convenience API while avoiding an eager import cycle.  The
# dashboard and quality-report modules depend on the low-level API above; in
# turn, importing them from here would make package initialization walk back
# through ``nlpie._api``.  PEP 562 lets these optional, higher-level exports be
# loaded only when a caller actually requests them.
_LAZY_EXPORTS = {
    "Dashboard": (".backends", "Dashboard"),
    "PlotBackend": (".backends", "PlotBackend"),
    "PlotlyBackend": (".backends", "PlotlyBackend"),
    "DashboardBuilder": (".dashboard.builder", "DashboardBuilder"),
    "compare_and_plot_delta": (
        ".dashboard.comparison",
        "compare_and_plot_delta",
    ),
    "compare_and_plot_grouped_bar": (
        ".dashboard.comparison",
        "compare_and_plot_grouped_bar",
    ),
    "compare_and_plot_radar": (
        ".dashboard.comparison",
        "compare_and_plot_radar",
    ),
    "ReportExporter": (".export", "ReportExporter"),
    "HtmlExporter": (".export", "HtmlExporter"),
    "JsonExporter": (".export", "JsonExporter"),
    "MarkdownExporter": (".export", "MarkdownExporter"),
    "Explanation": (".interpret", "Explanation"),
    "ExplanationProvider": (".interpret", "ExplanationProvider"),
    "ExplanationRegistry": (".interpret", "ExplanationRegistry"),
    "HubInfo": (".interpret", "HubInfo"),
    "HubnessExplanation": (".interpret", "HubnessExplanation"),
    "InterpretationReport": (".interpret", "InterpretationReport"),
    "explain_hubness": (".interpret", "explain_hubness"),
    "ClusteringMetrics": (".metrics", "ClusteringMetrics"),
    "EmbeddingQualityReport": (".metrics", "EmbeddingQualityReport"),
    "GeometryMetrics": (".metrics", "GeometryMetrics"),
    "IntrinsicMetrics": (".metrics", "IntrinsicMetrics"),
    "ProjectionMetrics": (".metrics", "ProjectionMetrics"),
    "RetrievalMetrics": (".metrics", "RetrievalMetrics"),
    "compare_models": (".metrics", "compare_models"),
    "evaluate_embedding_quality": (".metrics", "evaluate_embedding_quality"),
}

_DASHBOARD_EXPORTS = {
    "plot_comparison_delta",
    "plot_comparison_grouped_bar",
    "plot_comparison_radar",
    "plot_correlation_heatmap",
    "plot_eigenvalue_scree",
    "plot_embedding_scatter",
    "plot_hubness_bar",
    "plot_hubness_histogram",
    "plot_projection_quality",
    "plot_quality_report",
    "plot_retrieval_metrics",
    "plot_silhouette",
    "plot_similarity_distribution",
    "plot_similarity_heatmap",
    "plot_similarity_to_mean",
}

_LAZY_EXPORT_NAMES = (
    "EmbeddingQualityReport",
    "IntrinsicMetrics",
    "ClusteringMetrics",
    "GeometryMetrics",
    "ProjectionMetrics",
    "RetrievalMetrics",
    "evaluate_embedding_quality",
    "compare_models",
    "Explanation",
    "ExplanationProvider",
    "ExplanationRegistry",
    "HubnessExplanation",
    "HubInfo",
    "InterpretationReport",
    "explain_hubness",
    "Dashboard",
    "PlotBackend",
    "PlotlyBackend",
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
    "ReportExporter",
    "HtmlExporter",
    "JsonExporter",
    "MarkdownExporter",
)


def __getattr__(name: str):
    """Load higher-level public exports on first access."""
    if name in _DASHBOARD_EXPORTS:
        module_name, attribute_name = "._dashboard", name
    else:
        try:
            module_name, attribute_name = _LAZY_EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


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
    "silhouette_score",
    "davies_bouldin_score",
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
    "NlpieError",
    "PreprocessingError",
] + list(_LAZY_EXPORT_NAMES)
