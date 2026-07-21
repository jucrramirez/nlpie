"""Structured storytelling data derived from interpretation explanations.

This is the single place where severity counting, grouping, headline phrasing,
and recommendation aggregation live. Renderers (the Plotly storytelling
figure and the native HTML report) consume :class:`StoryData` and only handle
presentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..thresholds import CRITICAL, INFO, SEVERITY_ORDER, WARNING
from .base import Explanation

OK = "ok"

_HEALTHY_HEADLINE = "All metrics look healthy — no issues detected."


@dataclass(frozen=True)
class StoryGroup:
    severity: str
    explanations: list[Explanation]


@dataclass(frozen=True)
class StoryData:
    headline: str
    headline_severity: str  # one of "ok", "critical", "warning", "info"
    n_critical: int
    n_warning: int
    n_info: int
    groups: list[StoryGroup]
    recommendations: list[str]

    @property
    def is_healthy(self) -> bool:
        return self.headline_severity == OK


def build_story(explanations: Sequence[Explanation]) -> StoryData:
    """Aggregate explanations into presentation-ready storytelling data."""
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for exp in explanations:
        if exp.severity in counts:
            counts[exp.severity] += 1

    n_crit = counts[CRITICAL]
    n_warn = counts[WARNING]
    n_info = counts[INFO]

    if not explanations:
        headline = _HEALTHY_HEADLINE
        headline_severity = OK
    else:
        parts = []
        if n_crit:
            parts.append(f"{n_crit} critical")
        if n_warn:
            parts.append(f"{n_warn} warning")
        if n_info:
            parts.append(f"{n_info} info")
        headline = f"Detected {', '.join(parts)} issue(s) in the embedding space."
        headline_severity = CRITICAL if n_crit else (WARNING if n_warn else INFO)

    groups = [
        StoryGroup(
            severity=severity,
            explanations=[e for e in explanations if e.severity == severity],
        )
        for severity in SEVERITY_ORDER
        if counts[severity] > 0
    ]

    recommendations = [e.recommendation for e in explanations if e.recommendation]

    return StoryData(
        headline=headline,
        headline_severity=headline_severity,
        n_critical=n_crit,
        n_warning=n_warn,
        n_info=n_info,
        groups=groups,
        recommendations=recommendations,
    )
