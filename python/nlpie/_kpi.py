"""KPI card data derived from an embedding quality report.

Single source of KPI label/value/severity data, consumed by both the Plotly
KPI figure and the native HTML KPI cards. Colors are resolved by renderers
via :data:`nlpie.thresholds.SEVERITY_COLORS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .thresholds import (
    ARI_CRITICAL,
    ARI_WARNING,
    EFFECTIVE_RANK_CRITICAL_FRAC,
    EFFECTIVE_RANK_WARNING_FRAC,
    HUBNESS_CRITICAL,
    HUBNESS_WARNING,
    INFO,
    MEAN_SIMILARITY_CRITICAL,
    MEAN_SIMILARITY_WARNING,
    PROJECTION_CRITICAL,
    PROJECTION_WARNING,
    RETRIEVAL_CRITICAL,
    RETRIEVAL_WARNING,
    SILHOUETTE_CRITICAL,
    SILHOUETTE_WARNING,
    SIMILARITY_STD_WARNING,
    WARNING,
    classify_higher_is_worse,
    classify_lower_is_worse,
)


@dataclass(frozen=True)
class KpiCard:
    label: str
    value: str
    severity: str


def kpi_cards_data(report) -> list[KpiCard]:
    """Build the KPI card list for a report (empty when nothing is populated)."""
    n_dims = getattr(report, "n_dims", 0)
    cards: list[KpiCard] = []

    m = getattr(report, "intrinsic", None)
    if m is not None:
        cards.append(
            KpiCard(
                "Mean",
                f"{m.mean:.3f}",
                classify_higher_is_worse(m.mean, MEAN_SIMILARITY_CRITICAL, MEAN_SIMILARITY_WARNING),
            )
        )
        cards.append(
            KpiCard(
                "Std",
                f"{m.std:.3f}",
                WARNING if m.std < SIMILARITY_STD_WARNING else INFO,
            )
        )

    m = getattr(report, "geometry", None)
    if m is not None:
        cards.append(
            KpiCard(
                "Eff-Rank",
                f"{m.effective_rank:.1f}",
                classify_lower_is_worse(
                    m.effective_rank,
                    n_dims * EFFECTIVE_RANK_CRITICAL_FRAC,
                    n_dims * EFFECTIVE_RANK_WARNING_FRAC,
                ),
            )
        )
        cards.append(
            KpiCard(
                "Hubness",
                f"{m.hubness_skewness:.2f}",
                classify_higher_is_worse(m.hubness_skewness, HUBNESS_CRITICAL, HUBNESS_WARNING),
            )
        )

    m = getattr(report, "clustering", None)
    if m is not None:
        cards.append(
            KpiCard(
                "ARI",
                f"{m.ari:.3f}",
                classify_lower_is_worse(m.ari, ARI_CRITICAL, ARI_WARNING),
            )
        )
        cards.append(
            KpiCard(
                "Silhouette",
                f"{m.silhouette:.3f}",
                classify_lower_is_worse(m.silhouette, SILHOUETTE_CRITICAL, SILHOUETTE_WARNING),
            )
        )

    p = getattr(report, "projection", None)
    if p:
        avg_t = sum(x.trustworthiness for x in p) / len(p)
        avg_c = sum(x.continuity for x in p) / len(p)
        cards.append(
            KpiCard(
                "Trust",
                f"{avg_t:.3f}",
                classify_lower_is_worse(avg_t, PROJECTION_CRITICAL, PROJECTION_WARNING),
            )
        )
        cards.append(
            KpiCard(
                "Cont",
                f"{avg_c:.3f}",
                classify_lower_is_worse(avg_c, PROJECTION_CRITICAL, PROJECTION_WARNING),
            )
        )

    r = getattr(report, "retrieval", None)
    if r:
        avg_rec = sum(x.recall for x in r) / len(r)
        avg_ndcg = sum(x.ndcg for x in r) / len(r)
        cards.append(
            KpiCard(
                "Recall",
                f"{avg_rec:.3f}",
                classify_lower_is_worse(avg_rec, RETRIEVAL_CRITICAL, RETRIEVAL_WARNING),
            )
        )
        cards.append(
            KpiCard(
                "NDCG",
                f"{avg_ndcg:.3f}",
                classify_lower_is_worse(avg_ndcg, RETRIEVAL_CRITICAL, RETRIEVAL_WARNING),
            )
        )

    return cards
