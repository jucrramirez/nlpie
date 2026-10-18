"""Tests for the export module (HTML, JSON, Markdown)."""

import json

from nlpie.export import HtmlExporter, JsonExporter, MarkdownExporter
from nlpie.interpret import Explanation, InterpretationReport
from nlpie.metrics.quality import EmbeddingQualityReport, GeometryMetrics, IntrinsicMetrics


def _make_report() -> EmbeddingQualityReport:
    return EmbeddingQualityReport(
        model_name="test-model",
        n_samples=100,
        n_dims=64,
        intrinsic=IntrinsicMetrics(mean=0.5, std=0.1, min=0.2, max=0.8),
        geometry=GeometryMetrics(
            effective_rank=32.0,
            mean_similarity=0.5,
            hubness_skewness=0.3,
            hubness_k=5,
        ),
    )


class TestJsonExporter:
    def test_to_string_contains_keys(self):
        report = _make_report()
        exporter = JsonExporter()
        text = exporter.to_string(report)
        assert '"model_name"' in text
        assert '"test-model"' in text
        assert '"intrinsic"' in text
        assert '"geometry"' in text

    def test_export_writes_file(self, tmp_path):
        report = _make_report()
        exporter = JsonExporter()
        path = str(tmp_path / "report.json")
        exporter.export(report, path)
        with open(path) as f:
            data = f.read()
        assert "test-model" in data

    def test_empty_sections_omitted(self):
        report = EmbeddingQualityReport(model_name="empty", n_samples=0, n_dims=0)
        exporter = JsonExporter()
        data = json.loads(exporter.to_string(report))
        assert "intrinsic" not in data

    def test_interpretation_includes_detail_tier(self):
        """Regression (B11): the detail tier must reach the exports."""
        report = _make_report()
        interpretation = InterpretationReport(
            explanations=[
                Explanation(
                    metric="hubness",
                    severity="critical",
                    summary="Severe hubness.",
                    detail="A few points dominate all neighbour lists.",
                    recommendation="Apply whitening.",
                )
            ]
        )
        data = json.loads(JsonExporter().to_string(report, interpretation))
        assert data["interpretation"][0]["detail"] == "A few points dominate all neighbour lists."

    def test_default_output_contains_structured_presentation(self):
        report = _make_report()
        data = json.loads(JsonExporter().to_string(report))

        assert data["summary"]["status"] == "warning"
        assert "headline" in data["summary"]
        assert data["metric_interpretation"]
        assert data["next_steps"]
        assert data["data_availability"]["pairwise_similarities"] is False
        assert "<html>" not in JsonExporter().to_string(report)

    def test_explicit_interpretation_takes_precedence(self):
        report = _make_report()
        interpretation = InterpretationReport(
            explanations=[
                Explanation(
                    metric="custom",
                    severity="critical",
                    summary="Custom finding.",
                    detail="Custom detail.",
                    recommendation="Custom action.",
                )
            ]
        )

        data = json.loads(JsonExporter().to_string(report, interpretation))

        assert data["interpretation"][0]["metric"] == "custom"
        assert data["summary"]["status"] == "critical"
        assert data["next_steps"] == ["Custom action."]


class TestExportEscapes:
    def test_html_exporter_escapes_model_name(self):
        report = _make_report()
        report.model_name = "<script>alert(1)</script>"
        text = HtmlExporter().to_string(report)
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text


class TestMarkdownExporter:
    def test_to_string_contains_sections(self):
        report = _make_report()
        exporter = MarkdownExporter()
        text = exporter.to_string(report)
        assert "test-model" in text
        assert "Intrinsic geometry" in text
        assert "Geometry" in text

    def test_export_writes_file(self, tmp_path):
        report = _make_report()
        exporter = MarkdownExporter()
        path = str(tmp_path / "report.md")
        exporter.export(report, path)
        with open(path) as f:
            data = f.read()
        assert "Embedding Quality Report" in data

    def test_interpretation_includes_detail_tier(self):
        report = _make_report()
        interpretation = InterpretationReport(
            explanations=[
                Explanation(
                    metric="intrinsic",
                    severity="warning",
                    summary="High mean similarity.",
                    detail="Vectors share a dominant direction.",
                    recommendation="Center the embeddings.",
                )
            ]
        )
        text = MarkdownExporter().to_string(report, interpretation)
        assert "Vectors share a dominant direction." in text

    def test_default_output_contains_complete_readout(self):
        report = _make_report()
        text = MarkdownExporter().to_string(report)

        assert "## Executive Summary" in text
        assert "## Metric Interpretation" in text
        assert "## Recommended Next Steps" in text
        assert "## How to Read This Report" in text
        assert "Not measured" in text


class TestHtmlExporter:
    def test_to_string_contains_html(self):
        report = _make_report()
        exporter = HtmlExporter()
        text = exporter.to_string(report)
        assert "<html>" in text or "<!DOCTYPE" in text
        assert "test-model" in text

    def test_export_writes_file(self, tmp_path):
        report = _make_report()
        exporter = HtmlExporter()
        path = str(tmp_path / "report.html")
        exporter.export(report, path)
        with open(path) as f:
            data = f.read()
        assert len(data) > 0

    def test_default_output_contains_sample_sections_and_unavailable_states(self):
        report = _make_report()
        text = HtmlExporter().to_string(report)

        for section in (
            "Release readout",
            "Quality snapshot",
            "Evidence",
            "Metric interpretation",
            "Recommended next steps",
            "How to read this report",
        ):
            assert section in text
        assert "Not measured" in text
        assert "chart.js" in text.lower()
        assert ".hero-grid { grid-template-columns:minmax(0, 1fr);" in text
        assert "thresholdLines" in text
        assert '"warning":0.5' in text or '"warning": 0.5' in text
        assert '"critical":1' in text or '"critical": 1' in text

    def test_retrieval_html_contains_all_metrics(self):
        report = _make_report()
        from nlpie.metrics.quality import RetrievalMetrics

        report.retrieval = [RetrievalMetrics(1, 0.1, 0.2, 0.3, 0.4, 0.5)]
        text = HtmlExporter().to_string(report)

        for label in ("Recall", "Precision", "MRR", "nDCG", "Coverage"):
            assert label in text
