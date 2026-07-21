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

from collections.abc import Sequence
from dataclasses import dataclass, field

from nlpie._api import (
    _to_matrix,
    adjusted_rand_index,
    calinski_harabasz_score,
    compute_hubness,
    coverage_at_k,
    effective_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    normalized_mutual_info,
    pairwise_cosine_stats,
    precision_at_k,
    projection_quality,
    purity_score,
    recall_at_k,
    silhouette_score,
    similarity_to_global_mean,
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
    intrinsic: IntrinsicMetrics | None = None
    clustering: ClusteringMetrics | None = None
    geometry: GeometryMetrics | None = None
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
    embeddings_matrix: MatrixLike,
) -> tuple[IntrinsicMetrics, list[float]]:
    # Single Rust pass: returns the flattened upper triangle plus stats,
    # so no O(N²) matrix is materialized or re-extracted in Python.
    pairwise, mean, std, min_val, max_val = pairwise_cosine_stats(embeddings_matrix)
    return IntrinsicMetrics(mean=mean, std=std, min=min_val, max=max_val), pairwise


def _compute_clustering(
    embeddings_matrix: list[list[float]],
    labels: list[int],
    labels_pred: list[int] | None,
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


def _validate_inputs(
    n_samples: int,
    n_dims: int,
    hubness_k: int,
    low_dim: MatrixLike | None,
    projection_k_values: Sequence[int],
) -> None:
    """Fail fast with clear messages before any expensive computation."""
    if n_samples < 2:
        raise ValueError(
            f"evaluate_embedding_quality requires at least 2 samples, got {n_samples}."
        )
    if n_dims < 1:
        raise ValueError("Embeddings must have at least 1 dimension.")
    if not 1 <= hubness_k < n_samples:
        raise ValueError(
            f"hubness_k={hubness_k} is invalid for {n_samples} samples: "
            f"need 1 <= hubness_k < n_samples."
        )
    if low_dim is not None:
        low_mat = _to_matrix(low_dim)
        if len(low_mat) != n_samples:
            raise ValueError(
                f"low_dim has {len(low_mat)} rows but embeddings has {n_samples}; they must match."
            )
        if not projection_k_values:
            raise ValueError("projection_k_values must not be empty when low_dim is provided.")
        for k in projection_k_values:
            # Same domain as the Rust normalisation: the denominator
            # 2n - 3k - 1 must stay positive.
            if k < 1 or 2 * n_samples - 3 * k - 1 <= 0:
                raise ValueError(
                    f"projection k={k} is invalid for {n_samples} samples: "
                    f"need 1 <= k and 2*n_samples - 3*k - 1 > 0."
                )


def evaluate_embedding_quality(
    embeddings: MatrixLike,
    *,
    labels: Sequence[int] | None = None,
    labels_pred: Sequence[int] | None = None,
    hubness_k: int = 5,
    low_dim: MatrixLike | None = None,
    projection_k_values: Sequence[int] = (5, 10, 20),
    retrieved: Sequence[Sequence[int]] | None = None,
    relevant: Sequence[Sequence[int]] | None = None,
    retrieval_k_values: Sequence[int] = (1, 5, 10),
    model_name: str = "unnamed",
) -> tuple[EmbeddingQualityReport, InterpretationReport]:
    emb_mat = _to_matrix(embeddings)
    n_samples = len(emb_mat)
    n_dims = len(emb_mat[0]) if n_samples > 0 else 0

    _validate_inputs(n_samples, n_dims, hubness_k, low_dim, projection_k_values)

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
        int_labels = [int(label) for label in labels]
        int_pred = [int(label) for label in labels_pred] if labels_pred is not None else None
        report.clustering = _compute_clustering(emb_mat, int_labels, int_pred)

    if low_dim is not None:
        low_mat = _to_matrix(low_dim)
        # One Rust pass: both k-NN rank matrices are built once and reused
        # for every k value.
        for k, t, c in projection_quality(emb_mat, low_mat, projection_k_values):
            report.projection.append(ProjectionMetrics(k=k, trustworthiness=t, continuity=c))

    if retrieved is not None or relevant is not None:
        if retrieved is None or relevant is None:
            raise ValueError(
                "Both `retrieved` and `relevant` must be provided for retrieval metrics."
            )
        if len(retrieved) != len(relevant):
            raise ValueError(
                f"`retrieved` and `relevant` must have the same number of queries, "
                f"got {len(retrieved)} and {len(relevant)}."
            )
        if not retrieval_k_values:
            raise ValueError(
                "retrieval_k_values must not be empty when retrieval data is provided."
            )
        ret_lists = [list(r) for r in retrieved]
        rel_lists = [list(r) for r in relevant]
        n_queries = len(ret_lists)

        mrr_values = [mean_reciprocal_rank(ret_lists[i], rel_lists[i]) for i in range(n_queries)]
        mean_mrr = _mean(mrr_values)

        for k in retrieval_k_values:
            recall_vals = [recall_at_k(ret_lists[i], rel_lists[i], k) for i in range(n_queries)]
            prec_vals = [precision_at_k(ret_lists[i], rel_lists[i], k) for i in range(n_queries)]
            ndcg_vals = [ndcg_at_k(ret_lists[i], rel_lists[i], k) for i in range(n_queries)]
            cov = coverage_at_k(ret_lists, rel_lists, k)

            report.retrieval.append(
                RetrievalMetrics(
                    k=k,
                    recall=_mean(recall_vals),
                    precision=_mean(prec_vals),
                    mrr=mean_mrr,
                    ndcg=_mean(ndcg_vals),
                    coverage=cov,
                )
            )

    interpretation = ExplanationRegistry.explain_all(report)
    return report, interpretation


def compare_models(
    models: dict[str, MatrixLike],
    *,
    labels: Sequence[int] | None = None,
    labels_pred: Sequence[int] | None = None,
    hubness_k: int = 5,
    low_dims: dict[str, MatrixLike] | None = None,
    projection_k_values: Sequence[int] = (5, 10, 20),
    retrieved: dict[str, Sequence[Sequence[int]]] | None = None,
    relevant: Sequence[Sequence[int]] | None = None,
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
