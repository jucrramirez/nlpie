# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Added `python/metrics/quality.py` providing `EmbeddingQualityReport` and multi-model comparison to aggregate intrinsic, clustering, geometry, projection, and retrieval metrics.
- Added comprehensive Markdown API documentation under `docs/modules/` covering normalization, metrics, and quality reports.
- Added an end-to-end Jupyter notebook pipeline example at `python/examples/pipeline.ipynb`.
- Added clustering metrics: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Purity, Calinski‑Harabasz index, and Silhouette score.
- Added geometry and hubness diagnostics: effective rank, similarity to global mean, and hubness index (counts + skewness).
- **`FigureList`:** New `list` subclass returned by `DashboardBuilder.build()`. Wraps the figure list and provides a `.show()` method that shows all figures sequentially, plus `_ipython_display_()` for automatic rendering in Jupyter.
- **`DashboardBuilder`:** Added `max_heatmap_size` parameter (default 300) to `__init__` and `from_embeddings`. When the similarity matrix exceeds this size, it is evenly downsampled before rendering to prevent browser crashes.
- **Similarity heatmap in full dashboard:** `full_dashboard()` now includes a similarity heatmap chart (reconstructed from `pairwise_similarities`, downsampled to 300×300 max).
- **Similarity heatmap reconstruction:** `_reconstruct_similarity_matrix()` helper rebuilds the full N×N symmetric matrix from the 1D pairwise upper-triangle list, fixing a bug where a flat list was incorrectly passed to the heatmap function.

### Changed
- **Dashboard layout (Option 1):** `full_dashboard()` now returns `(fig_kpi, chart_list, fig_story)` where `chart_list` is a `list[tuple[str, go.Figure]]` of independent per-chart figures instead of a single `make_subplots` grid. This eliminates subplot overlap artifacts and enables per-chart personalization.
- **`DashboardBuilder.build()`:** Now returns `FigureList` (a `list` subclass) instead of plain `list`. Supports direct `.show()` chaining. The fallback path (no sections) also returns `FigureList`.
- **`DashboardBuilder.build()` heatmap fix:** The similarity heatmap section now properly reconstructs the 2D matrix from the 1D `pairwise_similarities` list using `_reconstruct_similarity_matrix()`. Previously it passed the flat list directly to `similarity_heatmap`, which expected a 2D matrix and could raise `TypeError` or render a broken figure.
- **`HtmlExporter`:** Charts are now laid out in a responsive `auto-fill` CSS Grid container (minimum 480px per column) instead of a fixed 2-column layout, accommodating the additional heatmap chart. KPI cards are rendered as native HTML/CSS elements (colored left bar, hover shadow). The Analysis & Recommendations section uses styled severity badges, grouped card layout, and a summary banner — no longer a plain Plotly annotation.
- **`ReportExporter` base class:** `to_string(report)` and `export(report, path)` now accept an optional `interpretation` keyword argument, enabling rich storytelling in HTML export.
- **Jupyter notebooks** (`pipeline.ipynb`, `pipeline_with_real_embeddings.ipynb`): Updated `DashboardBuilder` usage to chain `.show()` directly on `build()` instead of iterating manually.

---

## [0.1.0] - 2026-07-05

### Added
- Core pipeline with embedding generation.
- Rust–Python interoperability using `pyo3` and `rust-numpy`.
- High-level Python API (`nlpie._api.EmbeddingPreprocessor`).
- Core vector normalization and ops in Rust (PCA whitening, L2 normalization, mean centering, and standardizing columns).
- Initial structure for embedding metrics and operations.

---
