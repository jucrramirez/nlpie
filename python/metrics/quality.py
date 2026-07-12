"""Embedding quality report and model comparison (TASK-007).

This module aggregates all nlpie metric categories into a single
:class:`EmbeddingQualityReport` and provides convenience functions for
single-model evaluation and multi-model side-by-side comparison.

Example usage::

    import numpy as np
    from python.metrics.quality import evaluate_embedding_quality, compare_models

    embeddings = np.random.randn(200, 64).astype(np.float32).tolist()
    labels = [i % 5 for i in range(200)]

    report = evaluate_embedding_quality(
        embeddings,
        labels=labels,
        hubness_k=5,
        model_name="my-model",
    )
    print(report)

    # Compare two models
    emb_a = np.random.randn(200, 64).astype(np.float32).tolist()
    emb_b = np.random.randn(200, 64).astype(np.float32).tolist()
    comparison = compare_models(
        {"baseline": emb_a, "finetuned": emb_b},
        labels=labels,
        hubness_k=5,
    )
    print(comparison)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from nlpie._api import (
    cosine_similarity_matrix,
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


# ---------------------------------------------------------------------------
# Metric sub-reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntrinsicMetrics:
    """Summary statistics of the pairwise cosine similarity distribution.

    Attributes:
        mean: Mean pairwise cosine similarity.
        std:  Standard deviation of pairwise cosine similarities.
        min:  Minimum pairwise cosine similarity.
        max:  Maximum pairwise cosine similarity (excluding self-similarity).
    """

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
    """Clustering quality metrics computed against ground-truth labels.

    Attributes:
        ari:              Adjusted Rand Index.
        nmi:              Normalized Mutual Information.
        purity:           Purity score.
        silhouette:       Mean Silhouette Coefficient.
        calinski_harabasz: Calinski-Harabasz index.
    """

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
    """Geometry and pathology diagnostics for the embedding space.

    Attributes:
        effective_rank:       Effective rank of the covariance matrix.
        mean_similarity:      Mean cosine similarity to the global centroid.
        hubness_skewness:     Skewness of the K-NN hubness distribution.
        hubness_k:            K value used for hubness computation.
    """

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
    """Projection quality metrics at a specific K value.

    Attributes:
        k:               Neighbourhood size.
        trustworthiness: Trustworthiness score.
        continuity:      Continuity score.
    """

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
    """Retrieval quality metrics at a specific K value.

    Attributes:
        k:         Cut-off rank.
        recall:    Mean Recall@K.
        precision: Mean Precision@K.
        mrr:       Mean Reciprocal Rank.
        ndcg:      Mean nDCG@K.
        coverage:  Coverage@K.
    """

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
    """Aggregated embedding quality report across all metric categories.

    Not all sections are always populated — each section is ``None`` when
    the required inputs (labels, projection, retrieval data) were not supplied.

    Attributes:
        model_name:  Human-readable model identifier.
        n_samples:   Number of embedding vectors evaluated.
        n_dims:      Dimensionality of the embedding vectors.
        intrinsic:   Cosine similarity distribution statistics.
        clustering:  Clustering metrics (requires ``labels``).
        geometry:    Geometry and pathology diagnostics.
        projection:  Projection quality (requires ``low_dim``).
        retrieval:   Retrieval metrics (requires ``retrieved`` / ``relevant``).
    """

    model_name: str = "unnamed"
    n_samples: int = 0
    n_dims: int = 0
    intrinsic: Optional[IntrinsicMetrics] = None
    clustering: Optional[ClusteringMetrics] = None
    geometry: Optional[GeometryMetrics] = None
    projection: list[ProjectionMetrics] = field(default_factory=list)
    retrieval: list[RetrievalMetrics] = field(default_factory=list)

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

def _compute_intrinsic(embeddings_matrix: list[list[float]]) -> IntrinsicMetrics:
    """Computes intrinsic cosine similarity statistics."""
    sim = cosine_similarity_matrix(embeddings_matrix)
    n = len(sim)

    # Collect off-diagonal similarities.
    values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(sim[i][j])

    if not values:
        return IntrinsicMetrics(mean=0.0, std=0.0, min=0.0, max=0.0)

    total = sum(values)
    count = len(values)
    mean = total / count
    variance = sum((v - mean) ** 2 for v in values) / count
    std = variance ** 0.5
    return IntrinsicMetrics(
        mean=mean,
        std=std,
        min=min(values),
        max=max(values),
    )


def _compute_clustering(
    embeddings_matrix: list[list[float]],
    labels: list[int],
    labels_pred: Optional[list[int]],
) -> ClusteringMetrics:
    """Computes clustering metrics.

    When ``labels_pred`` is None, metrics that compare two clusterings
    (ARI, NMI, Purity) use ``labels`` against itself (perfect score).
    """
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
) -> GeometryMetrics:
    """Computes geometry and pathology diagnostics."""
    eff_rank = effective_rank(embeddings_matrix)
    sims = similarity_to_global_mean(embeddings_matrix)
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    _, skewness = compute_hubness(embeddings_matrix, hubness_k)
    return GeometryMetrics(
        effective_rank=eff_rank,
        mean_similarity=mean_sim,
        hubness_skewness=skewness,
        hubness_k=hubness_k,
    )


def _mean(values: list[float]) -> float:
    """Returns the arithmetic mean of ``values``, or ``0.0`` if empty."""
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
) -> EmbeddingQualityReport:
    """Evaluates embedding quality across all applicable metric categories.

    Only the categories for which the required inputs are provided will be
    populated.  At minimum, ``embeddings`` must be supplied, which enables
    intrinsic and geometry metrics.

    Args:
        embeddings:           2D embedding matrix ``(n_samples, n_dims)``.
        labels:               Ground-truth cluster/class labels (enables clustering).
        labels_pred:          Predicted cluster labels for ARI/NMI/Purity comparison.
                              When ``None``, ``labels`` is compared against itself.
        hubness_k:            K value for hubness computation (default ``5``).
        low_dim:              Low-dimensional projection (enables projection metrics).
        projection_k_values:  K values for trustworthiness/continuity.
        retrieved:            Per-query ranked retrieval lists (enables retrieval metrics).
        relevant:             Per-query ground-truth relevant IDs.
        retrieval_k_values:   K values for retrieval metrics.
        model_name:           Human-readable identifier for the report header.

    Returns:
        An :class:`EmbeddingQualityReport` with all applicable sections populated.

    Raises:
        ValueError:  If ``retrieved`` is provided without ``relevant`` or vice-versa.
    """
    emb_mat = _to_matrix(embeddings)
    n_samples = len(emb_mat)
    n_dims = len(emb_mat[0]) if n_samples > 0 else 0

    report = EmbeddingQualityReport(
        model_name=model_name,
        n_samples=n_samples,
        n_dims=n_dims,
    )

    # Intrinsic metrics (always computed).
    report.intrinsic = _compute_intrinsic(emb_mat)

    # Geometry metrics (always computed).
    report.geometry = _compute_geometry(emb_mat, hubness_k)

    # Clustering metrics (only when labels provided).
    if labels is not None:
        int_labels = [int(l) for l in labels]
        int_pred = [int(l) for l in labels_pred] if labels_pred is not None else None
        report.clustering = _compute_clustering(emb_mat, int_labels, int_pred)

    # Projection metrics (only when low_dim provided).
    if low_dim is not None:
        low_mat = _to_matrix(low_dim)
        for k in projection_k_values:
            t = trustworthiness(emb_mat, low_mat, k)
            c = continuity(emb_mat, low_mat, k)
            report.projection.append(ProjectionMetrics(k=k, trustworthiness=t, continuity=c))

    # Retrieval metrics (only when both retrieved and relevant provided).
    if retrieved is not None or relevant is not None:
        if retrieved is None or relevant is None:
            raise ValueError(
                "Both `retrieved` and `relevant` must be provided for retrieval metrics."
            )
        ret_lists = [list(r) for r in retrieved]
        rel_lists = [list(r) for r in relevant]
        n_queries = len(ret_lists)

        # MRR is not k-dependent; compute once.
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

    return report


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
) -> list[EmbeddingQualityReport]:
    """Evaluates and compares multiple embedding models side-by-side.

    Returns one :class:`EmbeddingQualityReport` per model, in the order of
    the ``models`` dictionary keys.

    Args:
        models:               Mapping of model name → embedding matrix.
        labels:               Shared ground-truth labels (same across all models).
        labels_pred:          Shared predicted labels.
        hubness_k:            K for hubness computation.
        low_dims:             Mapping of model name → low-dim projection (optional).
        projection_k_values:  K values for projection quality.
        retrieved:            Mapping of model name → per-query retrieval lists (optional).
        relevant:             Shared ground-truth relevant IDs for retrieval.
        retrieval_k_values:   K values for retrieval metrics.

    Returns:
        A list of :class:`EmbeddingQualityReport`, one per model.
    """
    reports: list[EmbeddingQualityReport] = []
    for name, emb in models.items():
        low_dim = low_dims.get(name) if low_dims else None
        ret = retrieved.get(name) if retrieved else None
        report = evaluate_embedding_quality(
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
    return reports
