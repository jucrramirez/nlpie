from __future__ import annotations

from typing import Optional

from .base import ReportExporter


_CSS = """
<style>
* { box-sizing:border-box; margin:0; padding:0 }
body { background:#f0f2f5; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:#1a1a2e }
.dash-wrap { max-width:1400px; margin:0 auto; padding:24px }

.dash-section { background:#fff; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:24px; overflow:hidden }
.dash-section-header { padding:16px 20px 0 20px; font-size:15px; font-weight:600; color:#1a1a2e; display:flex; align-items:center; gap:8px }
.dash-section-header .sub { font-size:11px; font-weight:400; color:#888 }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:12px; padding:16px 20px 20px }
.kpi-card { background:#fff; border-radius:10px; padding:14px 12px; position:relative; overflow:hidden; box-shadow:0 0 0 1px rgba(0,0,0,0.06); transition:box-shadow .15s }
.kpi-card:hover { box-shadow:0 2px 12px rgba(0,0,0,0.10) }
.kpi-card .bar { position:absolute; left:0; top:4px; bottom:4px; width:4px; border-radius:0 3px 3px 0 }
.kpi-card .label { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px }
.kpi-card .value { font-size:24px; font-weight:700; color:#1a1a2e }

/* Storytelling */
.story-body { padding:12px 20px 20px }
.story-body p { font-size:12px; line-height:1.6; margin-bottom:6px }
.story-summary { border-radius:8px; padding:10px 14px; margin-bottom:14px; font-size:13px; font-weight:500 }
.story-summary.critical { border-left:4px solid #d62728; background:#fef6f6; color:#b01a1a }
.story-summary.warning  { border-left:4px solid #ff7f0e; background:#fffaf2; color:#b85c00 }
.story-summary.info     { border-left:4px solid #1f77b4; background:#f4f8fc; color:#1a5276 }
.story-summary.ok       { border-left:4px solid #2ca02c; background:#f6fcf6; color:#2d6e2d }
.severity-badge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.03em }
.severity-badge.critical { background:#d62728; color:#fff }
.severity-badge.warning  { background:#ff7f0e; color:#fff }
.severity-badge.info     { background:#1f77b4; color:#fff }
.story-group { margin-bottom:10px }
.story-group-header { font-size:12px; font-weight:600; margin-bottom:6px; display:flex; align-items:center; gap:6px }
.story-item { padding:6px 0 6px 12px; border-left:2px solid #e0e0e0; margin-bottom:4px }
.story-item .metric { font-weight:500; font-size:12px }
.story-item .summary { color:#555; font-size:11px; margin-top:1px }
.story-item .rec { color:#777; font-size:11px; margin-top:2px; padding-left:14px; display:inline-block }
.rec-list { margin-top:10px; padding:10px 14px; background:#fafafa; border-radius:8px }
.rec-list strong { font-size:12px }
.rec-list ol { margin-top:4px; padding-left:18px }
.rec-list li { margin-bottom:3px; font-size:11px; color:#444 }

/* Chart grid */
.chart-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(480px, 1fr)); gap:24px; padding:16px 20px 20px }
.chart-grid .js-plotly-plot, .chart-grid .plotly-graph-div { width:100% !important }
@media (max-width:900px) { .chart-grid { grid-template-columns:1fr } }
</style>
"""


def _build_kpi_html(report) -> str:
    from nlpie.backends.plotly import _kpi_cards_data
    cards = _kpi_cards_data(report)
    if not cards:
        return ""
    parts = ['<div class="kpi-grid">']
    for label, value, color in cards:
        parts.append(
            f'<div class="kpi-card">'
            f'<div class="bar" style="background:{color}"></div>'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'</div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def _build_story_html(explanations: list) -> str:
    if not explanations:
        return (
            '<div class="story-body">'
            '<div class="story-summary ok">All metrics look healthy — no issues detected.</div>'
            '</div>'
        )

    n_crit = sum(1 for e in explanations if getattr(e, "severity", "") == "critical")
    n_warn = sum(1 for e in explanations if getattr(e, "severity", "") == "warning")
    n_info = sum(1 for e in explanations if getattr(e, "severity", "") == "info")
    total = n_crit + n_warn + n_info

    if total == 0:
        summary_class = "ok"
        summary_text = "All metrics look healthy — no issues detected."
    else:
        counts = ", ".join(
            s for s in [
                f"{n_crit} critical" if n_crit else "",
                f"{n_warn} warning" if n_warn else "",
                f"{n_info} info" if n_info else "",
            ] if s
        )
        severity = "critical" if n_crit else ("warning" if n_warn else "info")
        summary_class = severity
        summary_text = f"Detected {counts} issue(s) in the embedding space."

    parts = ['<div class="story-body">']
    parts.append(f'<div class="story-summary {summary_class}">{summary_text}</div>')

    for severity, header in [
        ("critical", "Critical Issues"),
        ("warning", "Warnings"),
        ("info", "Info"),
    ]:
        items = [e for e in explanations if getattr(e, "severity", "") == severity]
        if not items:
            continue
        parts.append('<div class="story-group">')
        parts.append(
            f'<div class="story-group-header">'
            f'<span class="severity-badge {severity}">{severity}</span>'
            f'{header}</div>'
        )
        for exp in items:
            parts.append('<div class="story-item">')
            parts.append(f'<div class="metric">{getattr(exp, "metric", "")}</div>')
            parts.append(f'<div class="summary">{getattr(exp, "summary", "")}</div>')
            rec = getattr(exp, "recommendation", "") or ""
            if rec:
                parts.append(f'<div class="rec">→ {rec}</div>')
            parts.append("</div>")
        parts.append("</div>")

    recs = [
        getattr(e, "recommendation", "")
        for e in explanations
        if getattr(e, "recommendation", "")
    ]
    if recs:
        parts.append('<div class="rec-list"><strong>Recommended Actions</strong><ol>')
        for rec in recs:
            parts.append(f"<li>{rec}</li>")
        parts.append("</ol></div>")

    parts.append("</div>")
    return "\n".join(parts)


class HtmlExporter(ReportExporter):
    def to_string(self, report, interpretation=None) -> str:
        try:
            from nlpie.backends.plotly import PlotlyBackend
            backend = PlotlyBackend()
            fig_kpi, chart_list, _ = backend.full_dashboard(report)

            parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>", _CSS]
            parts.append("<div class='dash-wrap'>")

            model_name = getattr(report, "model_name", "unnamed")
            n_samples = getattr(report, "n_samples", 0)
            n_dims = getattr(report, "n_dims", 0)

            # KPI — native HTML cards
            kpi_html = _build_kpi_html(report)
            parts.append("<div class='dash-section'>")
            parts.append(
                f"<div class='dash-section-header'>"
                f"Key Performance Indicators"
                f"<span class='sub'>{model_name} — {n_samples:,} samples · {n_dims:,} dimensions</span>"
                f"</div>"
            )
            parts.append(kpi_html)
            parts.append("</div>")

            # Charts grid
            if chart_list:
                parts.append("<div class='dash-section'>")
                parts.append(
                    f"<div class='dash-section-header'>"
                    f"Embedding Quality Charts"
                    f"<span class='sub'>{model_name}</span>"
                    f"</div>"
                )
                parts.append("<div class='chart-grid'>")
                for i, (_, fig) in enumerate(chart_list):
                    include_js = "cdn" if i == 0 else False
                    parts.append(fig.to_html(include_plotlyjs=include_js, full_html=False))
                parts.append("</div>")
                parts.append("</div>")

            # Storytelling — native HTML
            explanations = []
            if interpretation is not None:
                interp_explanations = getattr(interpretation, "explanations", None)
                if interp_explanations is not None:
                    explanations = interp_explanations
            if explanations:
                story_html = _build_story_html(explanations)
                parts.append("<div class='dash-section'>")
                parts.append(
                    f"<div class='dash-section-header'>"
                    f"Analysis &amp; Recommendations"
                    f"<span class='sub'>{model_name}</span>"
                    f"</div>"
                )
                parts.append(story_html)
                parts.append("</div>")

            parts.append("</div></body></html>")
            return "\n".join(parts)
        except ImportError:
            lines = [
                "<!DOCTYPE html><html><body>",
                "<h1>Embedding Quality Report</h1>",
                "<pre>",
                str(report),
                "</pre>",
                "<p><em>Plotly not installed — install with: pip install 'nlpie[plotting]'</em></p>",
                "</body></html>",
            ]
            return "\n".join(lines)

    def export(self, report, path: str, interpretation=None) -> None:
        html = self.to_string(report, interpretation=interpretation)
        with open(path, "w") as f:
            f.write(html)
