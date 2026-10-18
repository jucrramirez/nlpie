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
    from ._dashboard import (
        plot_comparison_delta as plot_comparison_delta,
    )
    from ._dashboard import (
        plot_comparison_grouped_bar as plot_comparison_grouped_bar,
    )
    from ._dashboard import (
        plot_comparison_radar as plot_comparison_radar,
    )
    from ._dashboard import (
        plot_correlation_heatmap as plot_correlation_heatmap,
    )
    from ._dashboard import (
        plot_eigenvalue_scree as plot_eigenvalue_scree,
    )
    from ._dashboard import (
        plot_embedding_scatter as plot_embedding_scatter,
    )
    from ._dashboard import (
        plot_hubness_bar as plot_hubness_bar,
    )
    from ._dashboard import (
        plot_hubness_histogram as plot_hubness_histogram,
    )
    from ._dashboard import (
        plot_projection_quality as plot_projection_quality,
    )
    from ._dashboard import (
        plot_quality_report as plot_quality_report,
    )
    from ._dashboard import (
        plot_retrieval_metrics as plot_retrieval_metrics,
    )
    from ._dashboard import (
        plot_silhouette as plot_silhouette,
    )
    from ._dashboard import (
        plot_similarity_distribution as plot_similarity_distribution,
    )
    from ._dashboard import (
        plot_similarity_heatmap as plot_similarity_heatmap,
    )
    from ._dashboard import (
        plot_similarity_to_mean as plot_similarity_to_mean,
    )
    from .backends import Dashboard as Dashboard
    from .backends import PlotBackend as PlotBackend
    from .backends import PlotlyBackend as PlotlyBackend
    from .dashboard.builder import DashboardBuilder as DashboardBuilder
    from .dashboard.comparison import (
        compare_and_plot_delta as compare_and_plot_delta,
    )
    from .dashboard.comparison import (
        compare_and_plot_grouped_bar as compare_and_plot_grouped_bar,
    )
    from .dashboard.comparison import (
        compare_and_plot_radar as compare_and_plot_radar,
    )
    from .export import HtmlExporter as HtmlExporter
    from .export import JsonExporter as JsonExporter
    from .export import MarkdownExporter as MarkdownExporter
    from .export import ReportExporter as ReportExporter
    from .interpret import (
        Explanation as Explanation,
    )
    from .interpret import (
        ExplanationProvider as ExplanationProvider,
    )
    from .interpret import (
        ExplanationRegistry as ExplanationRegistry,
    )
    from .interpret import (
        HubInfo as HubInfo,
    )
    from .interpret import (
        HubnessExplanation as HubnessExplanation,
    )
    from .interpret import (
        InterpretationReport as InterpretationReport,
    )
    from .interpret import (
        explain_hubness as explain_hubness,
    )
    from .metrics import (
        ClusteringMetrics as ClusteringMetrics,
    )
    from .metrics import (
        EmbeddingQualityReport as EmbeddingQualityReport,
    )
    from .metrics import (
        GeometryMetrics as GeometryMetrics,
    )
    from .metrics import (
        IntrinsicMetrics as IntrinsicMetrics,
    )
    from .metrics import (
        ProjectionMetrics as ProjectionMetrics,
    )
    from .metrics import (
        RetrievalMetrics as RetrievalMetrics,
    )
    from .metrics import (
        compare_models as compare_models,
    )
    from .metrics import (
        evaluate_embedding_quality as evaluate_embedding_quality,
    )

# Keep the public convenience API while avoiding an eager import cycle.  The
# dashboard and quality-report modules depend on the low-level API above; in
# turn, importing them from here would make package initialization walk back
# through ``nlpie._api``.  PEP 562 lets these optional, higher-level exports be
# loaded only when a caller actually requests them.
_LAZY_EXPORTS = {
    "Dashboard": ("nlpie.backends", "Dashboard"),
    "PlotBackend": ("nlpie.backends", "PlotBackend"),
    "PlotlyBackend": ("nlpie.backends", "PlotlyBackend"),
    "DashboardBuilder": ("nlpie.dashboard.builder", "DashboardBuilder"),
    "compare_and_plot_delta": (
        "nlpie.dashboard.comparison",
        "compare_and_plot_delta",
    ),
    "compare_and_plot_grouped_bar": (
        "nlpie.dashboard.comparison",
        "compare_and_plot_grouped_bar",
    ),
    "compare_and_plot_radar": (
        "nlpie.dashboard.comparison",
        "compare_and_plot_radar",
    ),
    "ReportExporter": ("nlpie.export", "ReportExporter"),
    "HtmlExporter": ("nlpie.export", "HtmlExporter"),
    "JsonExporter": ("nlpie.export", "JsonExporter"),
    "MarkdownExporter": ("nlpie.export", "MarkdownExporter"),
    "Explanation": ("nlpie.interpret", "Explanation"),
    "ExplanationProvider": ("nlpie.interpret", "ExplanationProvider"),
    "ExplanationRegistry": ("nlpie.interpret", "ExplanationRegistry"),
    "HubInfo": ("nlpie.interpret", "HubInfo"),
    "HubnessExplanation": ("nlpie.interpret", "HubnessExplanation"),
    "InterpretationReport": ("nlpie.interpret", "InterpretationReport"),
    "explain_hubness": ("nlpie.interpret", "explain_hubness"),
    "ClusteringMetrics": ("nlpie.metrics", "ClusteringMetrics"),
    "EmbeddingQualityReport": ("nlpie.metrics", "EmbeddingQualityReport"),
    "GeometryMetrics": ("nlpie.metrics", "GeometryMetrics"),
    "IntrinsicMetrics": ("nlpie.metrics", "IntrinsicMetrics"),
    "ProjectionMetrics": ("nlpie.metrics", "ProjectionMetrics"),
    "RetrievalMetrics": ("nlpie.metrics", "RetrievalMetrics"),
    "compare_models": ("nlpie.metrics", "compare_models"),
    "evaluate_embedding_quality": ("nlpie.metrics", "evaluate_embedding_quality"),
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
        module_name, attribute_name = "nlpie._dashboard", name
    else:
        try:
            module_name, attribute_name = _LAZY_EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
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
