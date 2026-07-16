"""Tests for the export module (HTML, JSON, Markdown)."""

from nlpie.metrics.quality import EmbeddingQualityReport, IntrinsicMetrics, GeometryMetrics
from nlpie.export import HtmlExporter, JsonExporter, MarkdownExporter


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
        text = exporter.to_string(report)
        assert '"intrinsic"' not in text


class TestMarkdownExporter:
    def test_to_string_contains_sections(self):
        report = _make_report()
        exporter = MarkdownExporter()
        text = exporter.to_string(report)
        assert "test-model" in text
        assert "Intrinsic Metrics" in text
        assert "Geometry & Pathology" in text

    def test_export_writes_file(self, tmp_path):
        report = _make_report()
        exporter = MarkdownExporter()
        path = str(tmp_path / "report.md")
        exporter.export(report, path)
        with open(path) as f:
            data = f.read()
        assert "Embedding Quality Report" in data


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
