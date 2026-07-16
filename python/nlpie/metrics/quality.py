"""Embedding quality report and model comparison (TASK-007).

This module aggregates all nlpie metric categories into a single
:class:`EmbeddingQualityReport` and provides convenience functions for
single-model evaluation and multi-model side-by-side comparison.

Example usage::

    import numpy as np
    from nlpie.metrics.quality import evaluate_embedding_quality, compare_models

    embeddings = np.random.randn(200, 64).astype(np.float32).tolist()
    labels = [i % 5 for i in range(200)]

    report, interpretation = evaluate_embedding_quality(
        embeddings,
        labels=labels,
        hubness_k=5,
        model_name="my-model",
    )
    print(report)
    print(interpretation)

    # Compare two models
    emb_a = np.random.randn(200, 64).astype(np.float32).tolist()
    emb_b = np.random.randn(200, 64).astype(np.float32).tolist()
    reports, interpretations = compare_models(
        {"baseline": emb_a, "finetuned": emb_b},
        labels=labels,
        hubness_k=5,
    )
    print(reports)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from nlpie._api import (
    cosine_similarity_matrix_stats,
    adjusted_rand_index,
    normalized_mutual_info,
    purity_score,
    calinski_harabasz_score,
    silhouette_score,
    effective_rank,
    similarity_to_global_mean,
    compute_hubness,
    trustworthiness,
    continuity,
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    coverage_at_k,
    _to_matrix,
)
from nlpie._types import MatrixLike
from nlpie.interpret import ExplanationRegistry, InterpretationReport


# ---------------------------------------------------------------------------
# Metric sub-reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntrinsicMetrics:
    mean: float
    std: float
    min: float
    max: float

    def __str__(self) -> str:
        return (
            "Intrinsic (Cosine Similarity)\n"
            f"  mean={self.mean:.4f}  std={self.std:.4f}  "
            f"min={self.min:.4f}  max={self.max:.4f}"
        )


@dataclass(frozen=True)
class ClusteringMetrics:
    ari: float
    nmi: float
    purity: float
    silhouette: float
    calinski_harabasz: float

    def __str__(self) -> str:
        return (
            "Clustering\n"
            f"  ARI={self.ari:.4f}  NMI={self.nmi:.4f}  Purity={self.purity:.4f}\n"
            f"  Silhouette={self.silhouette:.4f}  "
            f"Calinski-Harabasz={self.calinski_harabasz:.2f}"
        )


@dataclass(frozen=True)
class GeometryMetrics:
    effective_rank: float
    mean_similarity: float
    hubness_skewness: float
    hubness_k: int

    def __str__(self) -> str:
        return (
            "Geometry & Pathology\n"
            f"  Effective Rank={self.effective_rank:.2f}  "
            f"Mean Sim-to-Mean={self.mean_similarity:.4f}\n"
            f"  Hubness Skewness={self.hubness_skewness:.4f} (k={self.hubness_k})"
        )


@dataclass(frozen=True)
class ProjectionMetrics:
    k: int
    trustworthiness: float
    continuity: float

    def __str__(self) -> str:
        return (
            f"  k={self.k:>3}  "
            f"Trustworthiness={self.trustworthiness:.4f}  "
            f"Continuity={self.continuity:.4f}"
        )


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    coverage: float

    def __str__(self) -> str:
        return (
            f"  k={self.k:>3}  "
            f"R@K={self.recall:.4f}  P@K={self.precision:.4f}  "
            f"MRR={self.mrr:.4f}  nDCG={self.ndcg:.4f}  "
            f"Cov={self.coverage:.4f}"
        )


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingQualityReport:
    model_name: str = "unnamed"
    n_samples: int = 0
    n_dims: int = 0
    intrinsic: Optional[IntrinsicMetrics] = None
    clustering: Optional[ClusteringMetrics] = None
    geometry: Optional[GeometryMetrics] = None
    projection: list[ProjectionMetrics] = field(default_factory=list)
    retrieval: list[RetrievalMetrics] = field(default_factory=list)
    pairwise_similarities: list[float] = field(default_factory=list)
    hubness_counts: list[int] = field(default_factory=list)
    sim_to_mean: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "=" * 60
        lines = [
            sep,
            f"Embedding Quality Report: {self.model_name}",
            f"  Samples={self.n_samples}  Dims={self.n_dims}",
            sep,
        ]
        if self.intrinsic is not None:
            lines.append(str(self.intrinsic))
        if self.clustering is not None:
            lines.append(str(self.clustering))
        if self.geometry is not None:
            lines.append(str(self.geometry))
        if self.projection:
            lines.append("Projection Quality")
            for p in self.projection:
                lines.append(str(p))
        if self.retrieval:
            lines.append("Retrieval Quality")
            for r in self.retrieval:
                lines.append(str(r))
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_intrinsic(
    embeddings_matrix: list[list[float]],
) -> tuple[IntrinsicMetrics, list[float]]:
    matrix, mean, std, min_val, max_val = cosine_similarity_matrix_stats(embeddings_matrix)
    n = len(matrix)
    pairwise = [
        matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return IntrinsicMetrics(mean=mean, std=std, min=min_val, max=max_val), pairwise


def _compute_clustering(
    embeddings_matrix: list[list[float]],
    labels: list[int],
    labels_pred: Optional[list[int]],
) -> ClusteringMetrics:
    pred = labels_pred if labels_pred is not None else labels
    ari = adjusted_rand_index(labels, pred)
    nmi = normalized_mutual_info(labels, pred)
    purity = purity_score(labels, pred)
    sil = silhouette_score(embeddings_matrix, labels)
    ch = calinski_harabasz_score(embeddings_matrix, labels)
    return ClusteringMetrics(
        ari=ari,
        nmi=nmi,
        purity=purity,
        silhouette=sil,
        calinski_harabasz=ch,
    )


def _compute_geometry(
    embeddings_matrix: list[list[float]],
    hubness_k: int,
) -> tuple[GeometryMetrics, list[int], list[float]]:
    eff_rank = effective_rank(embeddings_matrix)
    sims = similarity_to_global_mean(embeddings_matrix)
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    counts, skewness = compute_hubness(embeddings_matrix, hubness_k)
    return (
        GeometryMetrics(
            effective_rank=eff_rank,
            mean_similarity=mean_sim,
            hubness_skewness=skewness,
            hubness_k=hubness_k,
        ),
        counts,
        sims,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_embedding_quality(
    embeddings: MatrixLike,
    *,
    labels: Optional[Sequence[int]] = None,
    labels_pred: Optional[Sequence[int]] = None,
    hubness_k: int = 5,
    low_dim: Optional[MatrixLike] = None,
    projection_k_values: Sequence[int] = (5, 10, 20),
    retrieved: Optional[Sequence[Sequence[int]]] = None,
    relevant: Optional[Sequence[Sequence[int]]] = None,
    retrieval_k_values: Sequence[int] = (1, 5, 10),
    model_name: str = "unnamed",
) -> tuple[EmbeddingQualityReport, InterpretationReport]:
    emb_mat = _to_matrix(embeddings)
    n_samples = len(emb_mat)
    n_dims = len(emb_mat[0]) if n_samples > 0 else 0

    report = EmbeddingQualityReport(
        model_name=model_name,
        n_samples=n_samples,
        n_dims=n_dims,
    )

    intrinsic, pairwise = _compute_intrinsic(emb_mat)
    report.intrinsic = intrinsic
    report.pairwise_similarities = pairwise

    geometry, hubness_counts, sim_to_mean = _compute_geometry(emb_mat, hubness_k)
    report.geometry = geometry
    report.hubness_counts = hubness_counts
    report.sim_to_mean = sim_to_mean

    if labels is not None:
        int_labels = [int(l) for l in labels]
        int_pred = [int(l) for l in labels_pred] if labels_pred is not None else None
        report.clustering = _compute_clustering(emb_mat, int_labels, int_pred)

    if low_dim is not None:
        low_mat = _to_matrix(low_dim)
        for k in projection_k_values:
            t = trustworthiness(emb_mat, low_mat, k)
            c = continuity(emb_mat, low_mat, k)
            report.projection.append(ProjectionMetrics(k=k, trustworthiness=t, continuity=c))

    if retrieved is not None or relevant is not None:
        if retrieved is None or relevant is None:
            raise ValueError(
                "Both `retrieved` and `relevant` must be provided for retrieval metrics."
            )
        ret_lists = [list(r) for r in retrieved]
        rel_lists = [list(r) for r in relevant]
        n_queries = len(ret_lists)

        mrr_values = [
            mean_reciprocal_rank(ret_lists[i], rel_lists[i]) for i in range(n_queries)
        ]
        mean_mrr = _mean(mrr_values)

        for k in retrieval_k_values:
            recall_vals = [recall_at_k(ret_lists[i], rel_lists[i], k) for i in range(n_queries)]
            prec_vals = [precision_at_k(ret_lists[i], rel_lists[i], k) for i in range(n_queries)]
            ndcg_vals = [ndcg_at_k(ret_lists[i], rel_lists[i], k) for i in range(n_queries)]
            cov = coverage_at_k(ret_lists, rel_lists, k)

            report.retrieval.append(RetrievalMetrics(
                k=k,
                recall=_mean(recall_vals),
                precision=_mean(prec_vals),
                mrr=mean_mrr,
                ndcg=_mean(ndcg_vals),
                coverage=cov,
            ))

    interpretation = ExplanationRegistry.explain_all(report)
    return report, interpretation


def compare_models(
    models: dict[str, MatrixLike],
    *,
    labels: Optional[Sequence[int]] = None,
    labels_pred: Optional[Sequence[int]] = None,
    hubness_k: int = 5,
    low_dims: Optional[dict[str, MatrixLike]] = None,
    projection_k_values: Sequence[int] = (5, 10, 20),
    retrieved: Optional[dict[str, Sequence[Sequence[int]]]] = None,
    relevant: Optional[Sequence[Sequence[int]]] = None,
    retrieval_k_values: Sequence[int] = (1, 5, 10),
) -> tuple[list[EmbeddingQualityReport], list[InterpretationReport]]:
    reports: list[EmbeddingQualityReport] = []
    interpretations: list[InterpretationReport] = []
    for name, emb in models.items():
        low_dim = low_dims.get(name) if low_dims else None
        ret = retrieved.get(name) if retrieved else None
        report, interp = evaluate_embedding_quality(
            emb,
            labels=labels,
            labels_pred=labels_pred,
            hubness_k=hubness_k,
            low_dim=low_dim,
            projection_k_values=projection_k_values,
            retrieved=ret,
            relevant=relevant,
            retrieval_k_values=retrieval_k_values,
            model_name=name,
        )
        reports.append(report)
        interpretations.append(interp)
    return reports, interpretations
