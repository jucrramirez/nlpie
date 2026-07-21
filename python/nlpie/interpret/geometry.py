from __future__ import annotations

from ..thresholds import (
    EFFECTIVE_RANK_CRITICAL_FRAC,
    EFFECTIVE_RANK_WARNING_FRAC,
    MEAN_SIMILARITY_CRITICAL,
    MEAN_SIMILARITY_WARNING,
    classify_higher_is_worse,
)
from .base import Explanation, ExplanationProvider


class GeometryExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["effective_rank", "similarity_to_mean"]

    def explain(self, report) -> list[Explanation]:
        geometry = getattr(report, "geometry", None)
        if geometry is None:
            return []

        explanations = [
            self._explain_effective_rank(geometry.effective_rank, getattr(report, "n_dims", 0)),
            self._explain_similarity_to_mean(geometry.mean_similarity),
        ]
        return [e for e in explanations if e is not None]

    @staticmethod
    def _explain_effective_rank(eff_rank: float, n_dims: int) -> Explanation | None:
        if n_dims <= 0:
            return None

        if eff_rank < n_dims * EFFECTIVE_RANK_CRITICAL_FRAC:
            severity = "critical"
            summary = f"Very low effective rank ({eff_rank:.1f} of {n_dims} dimensions)."
            detail = (
                f"The embedding occupies a much lower intrinsic dimensionality "
                f"({eff_rank:.1f}) than its ambient dimension ({n_dims}). This "
                f"suggests severe redundancy — most dimensions are noise or highly "
                f"correlated."
            )
            recommendation = (
                "Consider PCA whitening or reducing embedding dimensionality. If "
                "using a decoder-based model, check for posterior collapse."
            )
        elif eff_rank < n_dims * EFFECTIVE_RANK_WARNING_FRAC:
            severity = "warning"
            summary = f"Low effective rank ({eff_rank:.1f} of {n_dims} dimensions)."
            detail = (
                f"The embedding occupies only {eff_rank:.1f} effective dimensions "
                f"out of {n_dims}. Some dimensions carry little signal."
            )
            recommendation = (
                "Consider dimensionality reduction (PCA) or review training for "
                "dimensional collapse."
            )
        else:
            severity = "info"
            summary = f"Effective rank is {eff_rank:.1f}."
            detail = "The embedding space uses its dimensions effectively."
            recommendation = "No action required."

        return Explanation(
            metric="effective_rank",
            severity=severity,
            summary=summary,
            detail=detail,
            recommendation=recommendation,
        )

    @staticmethod
    def _explain_similarity_to_mean(mean_similarity: float) -> Explanation:
        severity = classify_higher_is_worse(
            mean_similarity, MEAN_SIMILARITY_CRITICAL, MEAN_SIMILARITY_WARNING
        )
        if severity == "critical":
            summary = (
                f"Embeddings are extremely concentrated around the centroid "
                f"(mean similarity={mean_similarity:.3f})."
            )
            detail = (
                "Nearly all points align with the global mean direction, so the "
                "space carries little per-point information. This often accompanies "
                "severe anisotropy and hubness."
            )
            recommendation = (
                "Apply mean-centering or PCA whitening, and check the training "
                "objective for collapse toward a dominant direction."
            )
        elif severity == "warning":
            summary = (
                f"Embeddings lean strongly toward the centroid "
                f"(mean similarity={mean_similarity:.3f})."
            )
            detail = (
                "A substantial shared component dominates the space, reducing the "
                "discriminative power of individual vectors."
            )
            recommendation = "Consider mean-centering the embeddings before downstream use."
        else:
            summary = f"Similarity to the global mean is {mean_similarity:.3f}."
            detail = "The space is well spread around the centroid."
            recommendation = "No action required."

        return Explanation(
            metric="similarity_to_mean",
            severity=severity,
            summary=summary,
            detail=detail,
            recommendation=recommendation,
        )
