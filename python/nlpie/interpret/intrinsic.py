from __future__ import annotations

from typing import Optional

from .base import Explanation, ExplanationProvider


class IntrinsicExplanationProvider(ExplanationProvider):
    def metric_keys(self) -> list[str]:
        return ["intrinsic"]

    def explain(self, report) -> Optional[Explanation]:
        intrinsic = getattr(report, "intrinsic", None)
        if intrinsic is None:
            return None

        mean = intrinsic.mean
        std = intrinsic.std

        if mean > 0.8:
            severity = "critical"
            summary = f"Very high mean cosine similarity ({mean:.3f}) — points are nearly collinear."
            detail = "All embedding vectors point in almost the same direction. This suggests low expressiveness; the embedding space may be dominated by a single mode."
            recommendation = "Consider L2 normalization, centering, or re-training with a contrastive objective that encourages diversity."
        elif mean > 0.5:
            severity = "warning"
            summary = f"High mean cosine similarity ({mean:.3f}) — points are strongly aligned."
            detail = "Moderately high similarity indicates redundancy across vectors. Downstream tasks relying on discriminative features may see degraded performance."
            recommendation = "Apply centering or PCA whitening to reduce alignment. Alternatively, check if the training objective promotes uniformity."
        elif mean < 0.1:
            severity = "info"
            summary = f"Very low mean cosine similarity ({mean:.3f}) — points are nearly orthogonal."
            detail = "Near-orthogonal vectors indicate a high-capacity space with low redundancy."
            recommendation = "No action required. The space appears well-conditioned."
        else:
            severity = "info"
            summary = f"Mean cosine similarity is {mean:.3f} (std={std:.3f})."
            detail = "The similarity distribution is within a typical range."
            recommendation = "No action required."

        return Explanation(
            metric="intrinsic",
            severity=severity,
            summary=summary,
            detail=detail,
            recommendation=recommendation,
        )
