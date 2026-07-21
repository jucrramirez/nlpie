from __future__ import annotations

from html import escape

from .._kpi import kpi_cards_data
from ..interpret.story import StoryData, build_story
from .base import ReportExporter

_CSS = """
<style>
* { box-sizing:border-box; margin:0; padding:0 }
body {
  background:#f0f2f5;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#1a1a2e;
}
.dash-wrap { max-width:1400px; margin:0 auto; padding:24px }

.dash-section {
  background:#fff; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,0.08);
  margin-bottom:24px; overflow:hidden;
}
.dash-section-header {
  padding:16px 20px 0 20px; font-size:15px; font-weight:600; color:#1a1a2e;
  display:flex; align-items:center; gap:8px;
}
.dash-section-header .sub { font-size:11px; font-weight:400; color:#888 }

/* KPI cards */
.kpi-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));
  gap:12px; padding:16px 20px 20px;
}
.kpi-card {
  background:#fff; border-radius:10px; padding:14px 12px; position:relative;
  overflow:hidden; box-shadow:0 0 0 1px rgba(0,0,0,0.06); transition:box-shadow .15s;
}
.kpi-card:hover { box-shadow:0 2px 12px rgba(0,0,0,0.10) }
.kpi-card .bar {
  position:absolute; left:0; top:4px; bottom:4px; width:4px; border-radius:0 3px 3px 0;
}
.kpi-card .bar.critical { background:#d62728 }
.kpi-card .bar.warning  { background:#ff7f0e }
.kpi-card .bar.info     { background:#2ca02c }
.kpi-card .label {
  font-size:11px; color:#888; text-transform:uppercase;
  letter-spacing:.04em; margin-bottom:4px;
}
.kpi-card .value { font-size:24px; font-weight:700; color:#1a1a2e }

/* Storytelling */
.story-body { padding:12px 20px 20px }
.story-body p { font-size:12px; line-height:1.6; margin-bottom:6px }
.story-summary {
  border-radius:8px; padding:10px 14px; margin-bottom:14px;
  font-size:13px; font-weight:500;
}
.story-summary.critical { border-left:4px solid #d62728; background:#fef6f6; color:#b01a1a }
.story-summary.warning  { border-left:4px solid #ff7f0e; background:#fffaf2; color:#b85c00 }
.story-summary.info     { border-left:4px solid #1f77b4; background:#f4f8fc; color:#1a5276 }
.story-summary.ok       { border-left:4px solid #2ca02c; background:#f6fcf6; color:#2d6e2d }
.severity-badge {
  display:inline-block; padding:2px 10px; border-radius:10px; font-size:10px;
  font-weight:600; text-transform:uppercase; letter-spacing:.03em;
}
.severity-badge.critical { background:#d62728; color:#fff }
.severity-badge.warning  { background:#ff7f0e; color:#fff }
.severity-badge.info     { background:#1f77b4; color:#fff }
.story-group { margin-bottom:10px }
.story-group-header {
  font-size:12px; font-weight:600; margin-bottom:6px;
  display:flex; align-items:center; gap:6px;
}
.story-item { padding:6px 0 6px 12px; border-left:2px solid #e0e0e0; margin-bottom:4px }
.story-item.critical { border-left-color:#d62728 }
.story-item.warning  { border-left-color:#ff7f0e }
.story-item.info     { border-left-color:#1f77b4 }
.story-item .metric { font-weight:500; font-size:12px }
.story-item .summary { color:#555; font-size:11px; margin-top:1px }
.story-item .detail { color:#666; font-size:11px; margin-top:2px; line-height:1.5 }
.story-item .rec {
  color:#777; font-size:11px; margin-top:2px; padding-left:14px; display:inline-block;
}
.rec-list { margin-top:10px; padding:10px 14px; background:#fafafa; border-radius:8px }
.rec-list strong { font-size:12px }
.rec-list ol { margin-top:4px; padding-left:18px }
.rec-list li { margin-bottom:3px; font-size:11px; color:#444 }

/* Chart grid */
.chart-grid {
  display:grid; grid-template-columns:repeat(auto-fill, minmax(480px, 1fr));
  gap:24px; padding:16px 20px 20px;
}
.chart-grid .js-plotly-plot, .chart-grid .plotly-graph-div { width:100% !important }
@media (max-width:900px) { .chart-grid { grid-template-columns:1fr } }
</style>
"""


def _build_kpi_html(report) -> str:
    cards = kpi_cards_data(report)
    if not cards:
        return ""
    parts = ['<div class="kpi-grid">']
    for card in cards:
        parts.append(
            f'<div class="kpi-card">'
            f'<div class="bar {escape(card.severity)}"></div>'
            f'<div class="label">{escape(card.label)}</div>'
            f'<div class="value">{escape(card.value)}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_story_html(story: StoryData) -> str:
    if story.is_healthy:
        return (
            '<div class="story-body">'
            f'<div class="story-summary ok">{escape(story.headline)}</div>'
            "</div>"
        )

    parts = ['<div class="story-body">']
    parts.append(
        f'<div class="story-summary {escape(story.headline_severity)}">'
        f"{escape(story.headline)}</div>"
    )

    group_headers = {
        "critical": "Critical Issues",
        "warning": "Warnings",
        "info": "Info",
    }
    for group in story.groups:
        severity = group.severity
        parts.append('<div class="story-group">')
        parts.append(
            f'<div class="story-group-header">'
            f'<span class="severity-badge {escape(severity)}">{escape(severity)}</span>'
            f"{group_headers.get(severity, severity)}</div>"
        )
        for exp in group.explanations:
            parts.append(f'<div class="story-item {escape(severity)}">')
            parts.append(f'<div class="metric">{escape(exp.metric)}</div>')
            parts.append(f'<div class="summary">{escape(exp.summary)}</div>')
            detail = getattr(exp, "detail", "") or ""
            if detail:
                parts.append(f'<div class="detail">{escape(detail)}</div>')
            rec = getattr(exp, "recommendation", "") or ""
            if rec:
                parts.append(f'<div class="rec">→ {escape(rec)}</div>')
            parts.append("</div>")
        parts.append("</div>")

    if story.recommendations:
        parts.append('<div class="rec-list"><strong>Recommended Actions</strong><ol>')
        for rec in story.recommendations:
            parts.append(f"<li>{escape(rec)}</li>")
        parts.append("</ol></div>")

    parts.append("</div>")
    return "\n".join(parts)


class HtmlExporter(ReportExporter):
    def to_string(self, report, interpretation=None) -> str:
        try:
            from nlpie.backends.plotly import PlotlyBackend

            backend = PlotlyBackend()
            dashboard = backend.full_dashboard(report)

            parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>", _CSS]
            parts.append("<div class='dash-wrap'>")

            model_name = escape(str(getattr(report, "model_name", "unnamed")))
            n_samples = getattr(report, "n_samples", 0)
            n_dims = getattr(report, "n_dims", 0)

            # KPI — native HTML cards (shared data source with the Plotly figure)
            kpi_html = _build_kpi_html(report)
            parts.append("<div class='dash-section'>")
            parts.append(
                f"<div class='dash-section-header'>"
                f"Key Performance Indicators"
                f"<span class='sub'>{model_name} — "
                f"{n_samples:,} samples · {n_dims:,} dimensions</span>"
                f"</div>"
            )
            parts.append(kpi_html)
            parts.append("</div>")

            # Charts grid
            if dashboard.charts:
                parts.append("<div class='dash-section'>")
                parts.append(
                    f"<div class='dash-section-header'>"
                    f"Embedding Quality Charts"
                    f"<span class='sub'>{model_name}</span>"
                    f"</div>"
                )
                parts.append("<div class='chart-grid'>")
                for i, (_, fig) in enumerate(dashboard.charts):
                    include_js = "cdn" if i == 0 else False
                    parts.append(fig.to_html(include_plotlyjs=include_js, full_html=False))
                parts.append("</div>")
                parts.append("</div>")

            # Storytelling — native HTML (shared story data with the Plotly figure)
            explanations = []
            if interpretation is not None:
                interp_explanations = getattr(interpretation, "explanations", None)
                if interp_explanations is not None:
                    explanations = interp_explanations
            if explanations:
                parts.append("<div class='dash-section'>")
                parts.append(
                    f"<div class='dash-section-header'>"
                    f"Analysis &amp; Recommendations"
                    f"<span class='sub'>{model_name}</span>"
                    f"</div>"
                )
                parts.append(_build_story_html(build_story(explanations)))
                parts.append("</div>")

            parts.append("</div></body></html>")
            return "\n".join(parts)
        except ImportError:
            lines = [
                "<!DOCTYPE html><html><body>",
                "<h1>Embedding Quality Report</h1>",
                "<pre>",
                escape(str(report)),
                "</pre>",
                "<p><em>Plotly not installed — "
                "install with: pip install 'nlpie[plotting]'</em></p>",
                "</body></html>",
            ]
            return "\n".join(lines)

    def export(self, report, path: str, interpretation=None) -> None:
        html = self.to_string(report, interpretation=interpretation)
        with open(path, "w") as f:
            f.write(html)
