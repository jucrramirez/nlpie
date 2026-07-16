from __future__ import annotations

from typing import Optional

from .base import Explanation, ExplanationProvider


class GeometryExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["effective_rank", "similarity_to_mean"]

    def explain(self, report) -> Optional[Explanation]:
        geometry = getattr(report, "geometry", None)
        intrinsic = getattr(report, "intrinsic", None)
        if geometry is None:
            return None

        eff_rank = geometry.effective_rank
        n_dims = getattr(report, "n_dims", 0)

        if n_dims > 0 and eff_rank < n_dims * 0.3:
            severity = "critical"
            summary = f"Very low effective rank ({eff_rank:.1f} of {n_dims} dimensions)."
            detail = f"The embedding occupies a much lower intrinsic dimensionality ({eff_rank:.1f}) than its ambient dimension ({n_dims}). This suggests severe redundancy — most dimensions are noise or highly correlated."
            recommendation = "Consider PCA whitening or reducing embedding dimensionality. If using a decoder-based model, check for posterior collapse."
        elif n_dims > 0 and eff_rank < n_dims * 0.6:
            severity = "warning"
            summary = f"Low effective rank ({eff_rank:.1f} of {n_dims} dimensions)."
            detail = f"The embedding occupies only {eff_rank:.1f} effective dimensions out of {n_dims}. Some dimensions carry little signal."
            recommendation = "Consider dimensionality reduction (PCA) or review training for dimensional collapse."
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
