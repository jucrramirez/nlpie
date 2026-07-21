"""Single source of truth for metric severity thresholds.

Every interpretation provider and every dashboard KPI card classifies metric
values through this module, so adjusting a breakpoint here applies everywhere
consistently (explanations, storytelling, KPI colors, and HTML badges).
"""

from __future__ import annotations

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

SEVERITY_ORDER = (CRITICAL, WARNING, INFO)

SEVERITY_COLORS = {
    CRITICAL: "#d62728",
    WARNING: "#ff7f0e",
    INFO: "#2ca02c",
}

# --- Hubness: skewness of the k-NN occurrence distribution (higher = worse)
HUBNESS_CRITICAL = 1.0
HUBNESS_WARNING = 0.5
# Below this skewness the neighbour-occurrence distribution is healthy.
HUBNESS_LOW = 0.25

# --- Intrinsic: mean pairwise cosine similarity (higher = worse)
MEAN_SIMILARITY_CRITICAL = 0.8
MEAN_SIMILARITY_WARNING = 0.5
# Below this, points are nearly orthogonal (notable, but not a pathology).
MEAN_SIMILARITY_NOTABLE_LOW = 0.1

# --- Intrinsic: std of pairwise similarities (narrower = suspicious)
SIMILARITY_STD_WARNING = 0.05

# --- Geometry: effective rank as a fraction of ambient dims (lower = worse)
EFFECTIVE_RANK_CRITICAL_FRAC = 0.3
EFFECTIVE_RANK_WARNING_FRAC = 0.6

# --- Clustering (lower = worse)
ARI_CRITICAL = 0.25
ARI_WARNING = 0.5
NMI_WARNING = 0.5
SILHOUETTE_CRITICAL = 0.1
SILHOUETTE_WARNING = 0.25

# --- Projection quality: trustworthiness / continuity (lower = worse)
PROJECTION_CRITICAL = 0.5
PROJECTION_WARNING = 0.8

# --- Retrieval quality: recall / nDCG (lower = worse)
RETRIEVAL_CRITICAL = 0.3
RETRIEVAL_WARNING = 0.7


def classify_higher_is_worse(value: float, critical: float, warning: float) -> str:
    """Classify a metric where larger values indicate a worse pathology."""
    if value > critical:
        return CRITICAL
    if value > warning:
        return WARNING
    return INFO


def classify_lower_is_worse(value: float, critical: float, warning: float) -> str:
    """Classify a metric where smaller values indicate a worse pathology."""
    if value < critical:
        return CRITICAL
    if value < warning:
        return WARNING
    return INFO


def worst_severity(severities: list[str]) -> str:
    """Return the most severe entry of ``severities`` (``INFO`` if empty)."""
    for severity in SEVERITY_ORDER:
        if severity in severities:
            return severity
    return INFO
