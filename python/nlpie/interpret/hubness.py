from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HubInfo:
    index: int
    count: int
    share_of_total: float


@dataclass(frozen=True)
class HubnessExplanation:
    summary: str
    severity: str
    skewness: float
    mean_count: float
    max_count: int
    top_hubs: list[HubInfo]
    interpretation: str


_INTERPRETATIONS = {
    "severe": "The K-NN distribution is heavily right-skewed — a small number of points (hubs) appear in the neighbour lists of a large fraction of other points. This can degrade retrieval quality and distort downstream evaluation.",
    "moderate": "The distribution shows moderate skew. Some points are acting as hubs; consider contrastive post-processing or examining the affected points if retrieval quality is a concern.",
    "low": "The embedding space is fairly well-behaved. Slight skew is present but unlikely to cause practical issues for most downstream tasks.",
    "none": "The embedding space shows no significant hubness pathology.",
}


def _classify_severity(
    skewness: float,
    severe_threshold: float = 1.0,
    moderate_threshold: float = 0.5,
    low_threshold: float = 0.25,
) -> tuple[str, str]:
    thresholds = [
        ("severe", severe_threshold),
        ("moderate", moderate_threshold),
        ("low", low_threshold),
    ]
    for severity, threshold in thresholds:
        if skewness > threshold:
            return severity, _INTERPRETATIONS[severity]
    return "none", _INTERPRETATIONS["none"]


def explain_hubness(
    counts: Sequence[int],
    skewness: float,
    k: int,
    *,
    severe_threshold: float = 1.0,
    moderate_threshold: float = 0.5,
    top_n: int = 5,
) -> HubnessExplanation:
    n_samples = len(counts)
    if n_samples == 0:
        raise ValueError("`counts` must contain at least one element")

    mean_count = float(k)
    max_count = max(counts)

    top_indices = sorted(
        range(n_samples), key=lambda i: counts[i], reverse=True
    )[:top_n]
    total_slots = n_samples * k
    top_hubs = [
        HubInfo(
            index=int(i),
            count=int(counts[i]),
            share_of_total=counts[i] / total_slots if total_slots > 0 else 0.0,
        )
        for i in top_indices
        if counts[i] > 0
    ]

    severity, interpretation = _classify_severity(
        skewness,
        severe_threshold=severe_threshold,
        moderate_threshold=moderate_threshold,
    )

    n_hubs = sum(1 for c in counts if c > k * 2)
    if n_hubs > 0 and severity in ("severe", "moderate"):
        summary = (
            f"{severity.capitalize()} hubness detected (skewness={skewness:.2f}). "
            f"{n_hubs} point(s) appear in more than twice the expected neighbour lists."
        )
    elif severity == "severe":
        summary = (
            f"Severe hubness detected (skewness={skewness:.2f}). "
            f"The K-NN distribution is heavily right-skewed."
        )
    elif severity == "moderate":
        summary = (
            f"Moderate hubness detected (skewness={skewness:.2f}). "
            f"Some points act as hubs."
        )
    elif severity == "low":
        summary = (
            f"Low hubness (skewness={skewness:.2f}). "
            f"The space is fairly well-behaved."
        )
    else:
        summary = (
            f"Negligible hubness (skewness={skewness:.2f}). "
            f"No significant hubness pathology detected."
        )

    return HubnessExplanation(
        summary=summary,
        severity=severity,
        skewness=float(skewness),
        mean_count=mean_count,
        max_count=max_count,
        top_hubs=top_hubs,
        interpretation=interpretation,
    )
