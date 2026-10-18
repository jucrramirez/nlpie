# ruff: noqa: E501

from __future__ import annotations

import json
import math
from html import escape
from typing import Any, cast

from ..thresholds import (
    EFFECTIVE_RANK_CRITICAL_FRAC,
    EFFECTIVE_RANK_WARNING_FRAC,
    HUBNESS_CRITICAL,
    HUBNESS_WARNING,
    PROJECTION_CRITICAL,
    PROJECTION_WARNING,
    RETRIEVAL_CRITICAL,
    RETRIEVAL_WARNING,
)
from .base import ReportExporter
from .presentation import MetricView, ReportPresentation, build_presentation

_CHART_JS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"

_CSS = """
<style>
:root {
  color-scheme: light;
  --bg: #f4f7f5; --surface: #ffffff; --surface-soft: #edf3ef;
  --text: #182019; --muted: #68736b; --border: #d9e2db;
  --primary: #206b50; --primary-soft: #dceee5; --critical: #b43f48;
  --critical-soft: #f8e5e7; --warning: #ad6b18; --warning-soft: #fff1d9;
  --healthy: #2d8154; --healthy-soft: #e2f2e7; --space: clamp(1rem, 2vw, 1.5rem);
}
:root[data-theme="dark"] { color-scheme: dark; --bg:#111613; --surface:#18211b; --surface-soft:#202c24;
  --text:#edf2ee; --muted:#aab7ad; --border:#304036; --primary:#7ed6ad; --primary-soft:#173d2e;
  --critical:#ff8d98; --critical-soft:#4c242b; --warning:#f6bb63; --warning-soft:#493317;
  --healthy:#83d8a5; --healthy-soft:#1b402a; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; overflow-x: clip; }
body { margin: 0; min-width: 0; background: var(--bg); color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5; }
a { color: var(--primary); }
button { font: inherit; }
.skip-link { position:absolute; left:-999px; top:auto; }
.skip-link:focus { left:1rem; top:1rem; z-index:10; background:var(--surface); padding:.6rem 1rem; border-radius:.5rem; }
.shell { width: min(100% - 2 * var(--space), 1180px); margin-inline:auto; min-width:0; }
.topbar { position:sticky; top:0; z-index:5; background:color-mix(in srgb, var(--surface) 94%, transparent);
  border-bottom:1px solid var(--border); backdrop-filter: blur(12px); }
.topbar-inner { min-height:3.5rem; display:flex; gap:1rem; align-items:center; justify-content:space-between; flex-wrap:wrap; }
.brand { font-weight:800; letter-spacing:-.02em; }
.topbar-actions { display:flex; align-items:center; gap:.8rem; flex-wrap:wrap; }
.topbar-actions a, .theme-toggle { border:1px solid var(--border); background:var(--surface); color:var(--text);
  border-radius:999px; padding:.35rem .75rem; text-decoration:none; font-size:.85rem; cursor:pointer; }
.report-header { padding:clamp(2.5rem, 7vw, 6rem) 0 2rem; }
.hero-grid, .decision-grid, .two-col, .metric-grid, .kpi-grid { display:grid; gap:var(--space); min-width:0; }
.hero-grid { grid-template-columns:minmax(0, 1fr); align-items:start; }
.eyebrow, .section-kicker { color:var(--primary); text-transform:uppercase; letter-spacing:.1em; font-size:.75rem; font-weight:800; }
h1, h2, h3, p { margin-top:0; }
h1 { margin-bottom:.8rem; max-width:18ch; font-size:clamp(2.1rem, 6vw, 4.4rem); line-height:1.02; letter-spacing:-.055em; }
h2 { font-size:clamp(1.45rem, 3vw, 2.15rem); letter-spacing:-.03em; margin-bottom:.45rem; }
h3 { font-size:1.05rem; margin-bottom:.35rem; }
.lead { max-width:68ch; color:var(--muted); font-size:clamp(1rem, 1.6vw, 1.2rem); }
.chips { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1.25rem; }
.chip, .status { border-radius:999px; padding:.35rem .7rem; font-size:.78rem; font-weight:750; background:var(--surface-soft); }
.card { min-width:0; background:var(--surface); border:1px solid var(--border); border-radius:1rem; box-shadow:0 8px 24px rgba(26,52,35,.05); }
.decision { padding:var(--space); }
.decision.critical { border-color:color-mix(in srgb, var(--critical) 45%, var(--border)); }
.decision.warning { border-color:color-mix(in srgb, var(--warning) 45%, var(--border)); }
.decision.healthy { border-color:color-mix(in srgb, var(--healthy) 45%, var(--border)); }
.decision-title { font-size:1.35rem; font-weight:800; margin:.45rem 0 .75rem; }
.decision ul, .simple-list { margin:.5rem 0 0; padding-left:1.2rem; }
.decision li, .simple-list li { margin:.45rem 0; color:var(--muted); }
.section { padding:1.5rem 0; }
.section-head { display:flex; justify-content:space-between; gap:1rem; align-items:end; flex-wrap:wrap; margin-bottom:1rem; }
.section-note { max-width:62ch; color:var(--muted); margin:0; }
.kpi-grid { grid-template-columns:repeat(4, minmax(0, 1fr)); }
.kpi { padding:1rem; }
.kpi .label { color:var(--muted); font-size:.78rem; font-weight:750; text-transform:uppercase; letter-spacing:.06em; }
.kpi .value { display:block; font-size:clamp(1.5rem, 3vw, 2.4rem); line-height:1.1; font-weight:850; margin:.35rem 0; }
.kpi .detail { color:var(--muted); font-size:.85rem; }
.critical, .status.critical { color:var(--critical); }
.warning, .status.warning { color:var(--warning); }
.healthy, .status.info { color:var(--healthy); }
.status { display:inline-block; background:var(--surface-soft); }
.status.critical { background:var(--critical-soft); } .status.warning { background:var(--warning-soft); }
.status.info { background:var(--healthy-soft); } .status.unavailable { color:var(--muted); }
.chart-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:var(--space); min-width:0; }
.chart-card { padding:var(--space); min-width:0; }
.chart-card.wide { grid-column:1 / -1; }
.chart-wrap { position:relative; width:100%; min-width:0; height:clamp(250px, 35vw, 360px); }
.chart-wrap.tall { height:clamp(280px, 42vw, 430px); }
.chart-wrap canvas { display:block; width:100% !important; height:100% !important; max-width:100%; }
.empty-state { min-height:clamp(150px, 20vw, 230px); display:grid; place-items:center; text-align:center; color:var(--muted); background:var(--surface-soft); border-radius:.7rem; padding:1rem; }
.narrative-grid, .steps-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:var(--space); }
.narrative, .step, .table-card { padding:var(--space); min-width:0; }
.step-number { color:var(--primary); font-size:1.5rem; font-weight:850; }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; min-width:680px; }
th, td { padding:.7rem .65rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
th { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
td { font-size:.88rem; }
td.explanation { color:var(--muted); max-width:35ch; }
details { padding:1rem var(--space); }
summary { cursor:pointer; font-weight:800; }
.footer { color:var(--muted); font-size:.82rem; padding:2rem 0 3rem; }
@media (max-width:900px) { .hero-grid, .kpi-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } .hero-grid { grid-template-columns:1fr; } }
@media (max-width:680px) { .shell { width:min(100% - 1.25rem, 1180px); } .chart-grid, .narrative-grid, .steps-grid, .kpi-grid { grid-template-columns:1fr; }
  .chart-card.wide { grid-column:auto; } h1 { max-width:none; } .topbar-inner { padding:.6rem 0; } }
</style>
"""


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, default=_json_default, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _histogram(values: list[float], bins: int = 16) -> dict[str, list[Any]]:
    if not values:
        return {"labels": [], "values": []}
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return {"labels": [f"{low:.2f}"], "values": [len(values)]}
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    labels = [f"{low + index * width:.2f}" for index in range(bins)]
    return {"labels": labels, "values": counts}


def _chart_data(report) -> dict[str, Any]:
    data: dict[str, Any] = {}
    retrieval = getattr(report, "retrieval", None) or []
    if retrieval:
        data["retrieval"] = {
            "labels": [f"K={item.k}" for item in retrieval],
            "datasets": [
                {"label": "Recall", "data": [item.recall * 100 for item in retrieval]},
                {"label": "Precision", "data": [item.precision * 100 for item in retrieval]},
                {"label": "MRR", "data": [item.mrr * 100 for item in retrieval]},
                {"label": "nDCG", "data": [item.ndcg * 100 for item in retrieval]},
                {"label": "Coverage", "data": [item.coverage * 100 for item in retrieval]},
            ],
            "thresholds": {
                "warning": RETRIEVAL_WARNING * 100,
                "critical": RETRIEVAL_CRITICAL * 100,
            },
            "threshold_unit": "%",
        }
    projection = getattr(report, "projection", None) or []
    if projection:
        data["projection"] = {
            "labels": [f"K={item.k}" for item in projection],
            "datasets": [
                {
                    "label": "Trustworthiness",
                    "data": [item.trustworthiness * 100 for item in projection],
                },
                {"label": "Continuity", "data": [item.continuity * 100 for item in projection]},
            ],
            "thresholds": {
                "warning": PROJECTION_WARNING * 100,
                "critical": PROJECTION_CRITICAL * 100,
            },
            "threshold_unit": "%",
        }
    geometry = getattr(report, "geometry", None)
    if geometry is not None:
        data["rank"] = {
            "labels": ["Effective rank", "Ambient dimensions"],
            "values": [geometry.effective_rank, report.n_dims],
            "thresholds": {
                "warning": report.n_dims * EFFECTIVE_RANK_WARNING_FRAC,
                "critical": report.n_dims * EFFECTIVE_RANK_CRITICAL_FRAC,
            },
        }
        data["hubness"] = {
            "labels": ["Observed skewness"],
            "values": [geometry.hubness_skewness],
            "thresholds": {"warning": HUBNESS_WARNING, "critical": HUBNESS_CRITICAL},
        }
    pairwise = getattr(report, "pairwise_similarities", None) or []
    if pairwise:
        data["similarity"] = _histogram([float(value) for value in pairwise])
    return data


def _metric_by_prefix(presentation: ReportPresentation, prefix: str) -> MetricView | None:
    return next(
        (
            metric
            for metric in presentation.metric_interpretation
            if metric.metric.startswith(prefix)
        ),
        None,
    )


def _metric_card(label: str, metric: MetricView | None, fallback: str) -> str:
    if metric is None:
        value, detail, status = "Not measured", fallback, "unavailable"
    else:
        value, detail, status = metric.formatted, metric.ideal, metric.status
    return (
        f'<article class="card kpi"><span class="label">{escape(label)}</span>'
        f'<strong class="value">{escape(value)}</strong>'
        f'<span class="status {escape(status)}">{escape(status.title())}</span>'
        f'<div class="detail">{escape(detail)}</div></article>'
    )


def _canvas_card(title: str, note: str, chart_id: str, wide: bool = False) -> str:
    return (
        f'<article class="card chart-card{" wide" if wide else ""}">'
        f'<h3>{escape(title)}</h3><p class="section-note">{escape(note)}</p>'
        f'<div class="chart-wrap"><canvas id="{escape(chart_id)}" role="img" aria-label="{escape(title)}"></canvas></div></article>'
    )


def _empty_card(title: str, reason: str) -> str:
    return f'<article class="card chart-card"><h3>{escape(title)}</h3><div class="empty-state"><span><strong>Not measured</strong><br>{escape(reason)}</span></div></article>'


def _build_html(report, presentation: ReportPresentation) -> str:
    model = escape(str(getattr(report, "model_name", "unnamed")))
    status = escape(presentation.status)
    metrics = presentation.metric_interpretation
    recall_candidates = [
        m for m in metrics if m.metric.startswith("retrieval.recall") and m.value is not None
    ]
    best_recall = (
        max(recall_candidates, key=lambda m: cast(float, m.value)) if recall_candidates else None
    )
    rows = []
    for metric in metrics:
        rows.append(
            f"<tr><td><strong>{escape(metric.metric)}</strong><br><small>{escape(metric.family)}</small></td>"
            f'<td>{escape(metric.formatted)}</td><td><span class="status {escape(metric.status)}">{escape(metric.status.title())}</span></td>'
            f'<td>{escape(metric.ideal)}</td><td>{escape(metric.reference)}</td><td class="explanation">{escape(metric.explanation)}</td></tr>'
        )
    blockers = "".join(f"<li>{escape(item)}</li>" for item in presentation.primary_blockers)
    healthy = "".join(f"<li>{escape(item)}</li>" for item in presentation.healthy_signals)
    actions = "".join(
        f'<article class="card step"><span class="step-number">{index:02d}</span><p>{escape(action)}</p></article>'
        for index, action in enumerate(presentation.next_steps, 1)
    )
    if not actions:
        actions = '<article class="card step"><span class="step-number">✓</span><p>No corrective action is required from the measured checks.</p></article>'
    chart_data = _chart_data(report)
    chart_cards = []
    chart_cards.append(
        _canvas_card(
            "Retrieval quality across K",
            "Higher is better. The dashed reference marks the existing critical Recall threshold.",
            "retrievalChart",
            wide=True,
        )
        if "retrieval" in chart_data
        else _empty_card("Retrieval quality across K", "No retrieval evaluation was supplied.")
    )
    chart_cards.append(
        _canvas_card(
            "Effective rank versus ambient dimensions",
            "Higher effective rank means more usable dimensions.",
            "rankChart",
        )
        if "rank" in chart_data
        else _empty_card("Effective rank versus ambient dimensions", "Geometry was not measured.")
    )
    chart_cards.append(
        _canvas_card(
            "Hubness skewness",
            "Near 0 is better; values above the reference indicate a long-tail neighbour problem.",
            "hubnessChart",
        )
        if "hubness" in chart_data
        else _empty_card("Hubness skewness", "Geometry was not measured.")
    )
    chart_cards.append(
        _canvas_card(
            "Projection fidelity",
            "Trustworthiness and continuity should be closer to 100%.",
            "projectionChart",
            wide=True,
        )
        if "projection" in chart_data
        else _empty_card("Projection fidelity", "No lower-dimensional projection was supplied.")
    )
    chart_cards.append(
        _canvas_card(
            "Pairwise similarity distribution",
            "Shows the spread of pairwise cosine similarities.",
            "similarityChart",
        )
        if "similarity" in chart_data
        else _empty_card(
            "Pairwise similarity distribution", "Raw pairwise similarities were not retained."
        )
    )
    availability_rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{'Available' if available else 'Not measured'}</td></tr>"
        for key, available in presentation.data_availability.items()
    )
    js = f"""
<script type="application/json" id="quality-report-chart-data">{_safe_json(chart_data)}</script>
<script src="{_CHART_JS}" defer></script>
<script>
document.addEventListener('DOMContentLoaded', function () {{
  const root = document.documentElement;
  const data = JSON.parse(document.getElementById('quality-report-chart-data').textContent);
  const toggle = document.querySelector('[data-theme-toggle]');
  let theme = window.localStorage.getItem('quality-report-theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const css = name => getComputedStyle(root).getPropertyValue(name).trim();
  const colors = ['#b43f48', '#ad6b18', '#206b50', '#7a5cff', '#2d8154'];
  const thresholdPlugin = {{
    id: 'thresholdLines',
    afterDraw(chart, args, pluginOptions) {{
      const thresholds = pluginOptions.thresholds || {{}};
      const y = chart.scales.y;
      const area = chart.chartArea;
      if (!y || !area) return;
      Object.entries(thresholds).forEach(([level, value]) => {{
        const yPos = y.getPixelForValue(value);
        if (!Number.isFinite(yPos) || yPos < area.top || yPos > area.bottom) return;
        const color = level === 'critical' ? '#b43f48' : '#ad6b18';
        const suffix = pluginOptions.suffix || '';
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        ctx.beginPath(); ctx.moveTo(area.left, yPos); ctx.lineTo(area.right, yPos); ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '600 11px Inter, sans-serif';
        ctx.fillText(level + ': ' + Number(value).toFixed(1) + suffix, area.left + 8, yPos - 6);
        ctx.restore();
      }});
    }}
  }};
  const common = (type, source, options = {{}}) => {{
    if (!source) return;
    const canvas = document.getElementById(options.id);
    if (!canvas || !window.Chart) return;
    const datasets = (source.datasets || []).map((item, index) => ({{
      ...item, borderColor: colors[index % colors.length], backgroundColor: colors[index % colors.length],
      borderWidth: 2, tension: .25, pointRadius: 3
    }}));
    new Chart(canvas, {{ type, data: {{ labels: source.labels, datasets }}, options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ usePointStyle: true, color: css('--text') }} }},
        thresholdLines: {{ thresholds: source.thresholds || {{}}, suffix: source.threshold_unit || '' }} }},
      scales: {{ x: {{ ticks: {{ color: css('--muted') }}, grid: {{ color: css('--border') }} }},
        y: {{ beginAtZero: true, max: options.max, ticks: {{ color: css('--muted') }}, grid: {{ color: css('--border') }} }} }},
      ...options.extra
    }}, plugins: [thresholdPlugin]}});
  }};
  const bar = (source, id, labels, max) => {{
    const canvas = document.getElementById(id); if (!canvas || !window.Chart) return;
    new Chart(canvas, {{
      type: 'bar',
      data: {{ labels, datasets: [{{ data: source.values,
        backgroundColor: ['#ad6b18', '#206b50'], borderColor: ['#ad6b18', '#206b50'],
        borderWidth: 1.5, borderRadius: 8 }}] }},
      options: {{
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {{
          legend: {{ display: false }},
          thresholdLines: {{ thresholds: source.thresholds || {{}}, suffix: source.threshold_unit || '' }}
        }},
        scales: {{
          x: {{ ticks: {{ color: css('--muted') }}, grid: {{ display: false }} }},
          y: {{ beginAtZero: true, max, ticks: {{ color: css('--muted') }}, grid: {{ color: css('--border') }} }}
        }}
      }},
      plugins: [thresholdPlugin]
    }});
  }};
  const render = () => {{
    root.setAttribute('data-theme', theme);
    if (toggle) toggle.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
    if (!window.Chart) return;
    common('line', data.retrieval, {{id:'retrievalChart', max:100}});
    common('line', data.projection, {{id:'projectionChart', max:100}});
    if (data.rank) bar(data.rank, 'rankChart', data.rank.labels, Math.max(...data.rank.values) * 1.15);
    if (data.hubness) bar(data.hubness, 'hubnessChart', data.hubness.labels, Math.max(2, data.hubness.values[0] * 1.25));
    if (data.similarity) bar(data.similarity, 'similarityChart', data.similarity.labels, Math.max(...data.similarity.values) * 1.15);
  }};
  if (toggle) toggle.addEventListener('click', () => {{
    theme = theme === 'dark' ? 'light' : 'dark';
    window.localStorage.setItem('quality-report-theme', theme);
    window.location.reload();
  }});
  const waitForChart = () => window.Chart ? render() : window.setTimeout(waitForChart, 25); waitForChart();
}});
</script>
"""
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>Embedding Quality Report — {model}</title>{_CSS}</head><body>",
            '<a class="skip-link" href="#main">Skip to report</a>',
            '<header class="topbar"><div class="shell topbar-inner"><span class="brand">Embedding Quality Report</span><div class="topbar-actions"><a href="#evidence">Jump to evidence</a><button class="theme-toggle" data-theme-toggle type="button">Toggle theme</button></div></div></header>',
            '<main id="main" class="shell">',
            f'<header class="report-header"><div class="hero-grid"><div><div class="eyebrow">{model} · {report.n_samples:,} samples · {report.n_dims:,} dimensions</div><h1>{escape(presentation.headline)}</h1><p class="lead">{escape(presentation.why_it_matters)}</p><div class="chips"><span class="chip">Status: {escape(status.title())}</span><span class="chip">Measured metrics: {len(metrics)}</span></div></div>',
            f'<aside class="card decision {status}" aria-label="Release readout"><span class="eyebrow">Release readout</span><div class="decision-title">{escape(presentation.headline)}</div><ul>{blockers or "<li>No critical or warning findings were identified.</li>"}{healthy or ""}</ul></aside></div></header>',
            '<section class="section" aria-labelledby="snapshot-title"><div class="section-head"><div><div class="section-kicker">At a glance</div><h2 id="snapshot-title">Quality snapshot</h2></div><p class="section-note">Each card states what the number means and which direction is preferred.</p></div><div class="kpi-grid">',
            _metric_card(
                "Best Recall",
                best_recall,
                "Retrieval was not measured.",
            ),
            _metric_card(
                "Cluster separation",
                _metric_by_prefix(presentation, "clustering.silhouette"),
                "Clustering was not measured.",
            ),
            _metric_card(
                "Projection fidelity",
                _metric_by_prefix(presentation, "projection.trustworthiness"),
                "Projection was not measured.",
            ),
            _metric_card(
                "Effective geometry",
                _metric_by_prefix(presentation, "geometry.effective_rank"),
                "Geometry was not measured.",
            ),
            "</div></section>",
            f'<section id="evidence" class="section" aria-labelledby="evidence-title"><div class="section-head"><div><div class="section-kicker">Evidence</div><h2 id="evidence-title">What the measurements show</h2></div><p class="section-note">Charts are responsive and use only values present in the report.</p></div><div class="chart-grid">{"".join(chart_cards)}</div></section>',
            '<section class="section" aria-labelledby="interpretation-title"><div class="section-head"><div><div class="section-kicker">Interpretation</div><h2 id="interpretation-title">What this means</h2></div></div><div class="narrative-grid"><article class="card narrative"><h3>Primary blockers</h3><ul class="simple-list">'
            + (blockers or "<li>None identified.</li>")
            + '</ul></article><article class="card narrative"><h3>Healthy signals</h3><ul class="simple-list">'
            + (healthy or "<li>No additional healthy signals reported.</li>")
            + '</ul></article><article class="card narrative"><h3>Why it matters</h3><p>'
            + escape(presentation.why_it_matters)
            + "</p></article></div></section>",
            f'<section class="section" aria-labelledby="steps-title"><div class="section-head"><div><div class="section-kicker">Action</div><h2 id="steps-title">Recommended next steps</h2></div></div><div class="steps-grid">{actions}</div></section>',
            '<section class="section" aria-labelledby="metrics-title"><div class="section-head"><div><div class="section-kicker">Reference</div><h2 id="metrics-title">Metric interpretation</h2></div><p class="section-note">Thresholds are guidance from the library; metrics without dedicated thresholds remain informational.</p></div><div class="card table-card"><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Observed</th><th>Status</th><th>Ideal direction</th><th>Reference</th><th>What it evaluates</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div></section>',
            f'<section class="section" aria-labelledby="coverage-title"><div class="section-head"><div><div class="section-kicker">Coverage</div><h2 id="coverage-title">What was measured</h2></div></div><div class="card table-card"><div class="table-wrap"><table><thead><tr><th>Evidence</th><th>Availability</th></tr></thead><tbody>{availability_rows}</tbody></table></div></div></section>',
            '<section class="section" aria-labelledby="read-title"><div class="card"><details open><summary id="read-title">How to read this report</summary><ul class="simple-list"><li><strong>Intrinsic geometry:</strong> asks whether vectors are too similar or point in one dominant direction.</li><li><strong>Clustering:</strong> asks whether groups are separated and internally coherent. Silhouette is higher-is-better; Davies–Bouldin is lower-is-better.</li><li><strong>Geometry:</strong> asks how many dimensions are useful and whether a few points dominate neighbour lists.</li><li><strong>Projection:</strong> asks whether a 2D or low-dimensional view preserves neighbourhood relationships.</li><li><strong>Retrieval:</strong> asks whether relevant items are found and ranked near the top. Higher values are better; 100% is the maximum.</li><li><strong>Not measured:</strong> means the required input or raw trace was not available, not that the metric failed.</li></ul></details></div></section>',
            f'<footer class="footer">Generated for {model}. Chart data and narrative content are derived from the supplied quality report.</footer>',
            "</main>",
            js,
            "</body></html>",
        ]
    )


class HtmlExporter(ReportExporter):
    def to_string(self, report, interpretation=None) -> str:
        return _build_html(report, build_presentation(report, interpretation))

    def export(self, report, path: str, interpretation=None) -> None:
        with open(path, "w") as f:
            f.write(self.to_string(report, interpretation=interpretation))
