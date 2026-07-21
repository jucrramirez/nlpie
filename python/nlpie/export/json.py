from __future__ import annotations

import json

from .base import ReportExporter


def _report_to_dict(report) -> dict:
    data = {
        "model_name": report.model_name,
        "n_samples": report.n_samples,
        "n_dims": report.n_dims,
    }

    if hasattr(report, "intrinsic") and report.intrinsic is not None:
        m = report.intrinsic
        data["intrinsic"] = {"mean": m.mean, "std": m.std, "min": m.min, "max": m.max}

    if hasattr(report, "clustering") and report.clustering is not None:
        m = report.clustering
        data["clustering"] = {
            "ari": m.ari,
            "nmi": m.nmi,
            "purity": m.purity,
            "silhouette": m.silhouette,
            "calinski_harabasz": m.calinski_harabasz,
        }

    if hasattr(report, "geometry") and report.geometry is not None:
        m = report.geometry
        data["geometry"] = {
            "effective_rank": m.effective_rank,
            "mean_similarity": m.mean_similarity,
            "hubness_skewness": m.hubness_skewness,
            "hubness_k": m.hubness_k,
        }

    if hasattr(report, "projection") and report.projection:
        data["projection"] = [
            {"k": p.k, "trustworthiness": p.trustworthiness, "continuity": p.continuity}
            for p in report.projection
        ]

    if hasattr(report, "retrieval") and report.retrieval:
        data["retrieval"] = [
            {
                "k": r.k,
                "recall": r.recall,
                "precision": r.precision,
                "mrr": r.mrr,
                "ndcg": r.ndcg,
                "coverage": r.coverage,
            }
            for r in report.retrieval
        ]

    return data


class JsonExporter(ReportExporter):
    def __init__(self, indent: int = 2):
        self.indent = indent

    def to_string(self, report, interpretation=None) -> str:
        data = _report_to_dict(report)
        if interpretation is not None:
            explanations = getattr(interpretation, "explanations", None)
            if explanations:
                data["interpretation"] = [
                    {
                        "metric": e.metric,
                        "severity": e.severity,
                        "summary": e.summary,
                        "detail": getattr(e, "detail", ""),
                        "recommendation": getattr(e, "recommendation", ""),
                    }
                    for e in explanations
                ]
        return json.dumps(data, indent=self.indent)

    def export(self, report, path: str, interpretation=None) -> None:
        with open(path, "w") as f:
            f.write(self.to_string(report, interpretation=interpretation))
