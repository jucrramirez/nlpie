# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.2.0a1] - 2026-10-18

### Added
- Cosine distance support for clustering, hubness, and projection metrics, with cosine defaults for high-dimensional NLP embeddings.
- Cosine Davies–Bouldin clustering score and report output.

### Changed
- Projection metrics use cosine distance for high-dimensional embeddings and Euclidean distance for low-dimensional coordinates by default.
- Clustering reports leave ARI, NMI, and purity unavailable until predicted labels are supplied.
- Report-level MRR is now calculated independently at every requested cutoff.

### Removed
- Calinski–Harabasz scoring from the public API and quality report because its Euclidean centroid assumptions do not match the documented directional NLP embedding use case.

### Fixed
- Hubness, silhouette, trustworthiness, and continuity no longer use raw Euclidean geometry by default on unnormalized NLP embeddings.
- Clustering reports no longer compare ground-truth labels with themselves when predictions are missing.

### Added
- **`Dashboard` dataclass:** `full_dashboard()` and `plot_quality_report()` now return a `Dashboard` (figures + titles with `.show()` / `.write_html()`), replacing the old 3-tuple.
- **`nlpie/thresholds.py`:** single source of truth for all severity breakpoints, shared by interpretation providers and KPI cards.
- **`interpret/story.py`:** `build_story()` produces severity-grouped `StoryData` consumed by both the Plotly dashboard and the HTML/Markdown exporters.
- **One-pass Rust `projection_quality(high, low, k_values)`:** computes trustworthiness + continuity for all k from a single pair of k-NN rank matrices.
- **`pairwise_cosine_stats(embeddings)`:** compact upper-triangle pairwise cosines plus mean/std/min/max without materialising the N×N matrix in Python.
- **`openblas-static` cargo feature:** statically compiled OpenBLAS fallback for machines without system BLAS (notably Windows).
- **Pre-commit config** (ruff, mypy, cargo fmt/clippy), **CI matrix** (Linux/macOS/Windows × Python 3.12/3.13) with wheel stub verification and PEP 517 install validation, and a **pytest-benchmark** suite under `benches/`.
- New tests: `test_comparison.py` (metric alignment, baseline validation), `test_interpret.py` (providers, thresholds consistency, story building), strict-validation and interpretation-integration tests in `test_quality_report.py`.
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
- **Strict input validation:** `evaluate_embedding_quality` validates shapes, k bounds (`1 <= k` and `2n - 3k - 1 > 0`), and retrieval inputs up front — no silent clamping. Degenerate k raises `PreprocessingError::InvalidK` in Rust and Python.
- **Comparison charts align by metric name:** models missing a section now get `None` placeholders (rendered as gaps) instead of silently shifted values; unknown `baseline` in `compare_and_plot_delta` raises `ValueError`; effective rank is plotted as a unit-normalized fraction.
- **Top-level exports:** the flagship API (`evaluate_embedding_quality`, `DashboardBuilder`, exporters, interpret types) is importable directly from `nlpie`.
- **Runtime dependencies reduced to `numpy` only**; maturin/pytest/pyarrow moved to the `dev` extra.
- **Interpretation providers return `list[Explanation]`** and flow end-to-end through `evaluate_embedding_quality`; all exporters render the explanation `detail` tier.
- **Performance:** Rayon-parallel silhouette score and k-NN rank construction; the Python layer no longer builds an O(N²) similarity matrix just for summary stats.
- **Dashboard layout (Option 1):** `full_dashboard()` now returns a `Dashboard` whose `chart_list`-style figures are independent per-chart figures instead of a single `make_subplots` grid. This eliminates subplot overlap artifacts and enables per-chart personalization.
- **`DashboardBuilder.build()`:** Now returns `FigureList` (a `list` subclass) instead of plain `list`. Supports direct `.show()` chaining. The fallback path (no sections) also returns `FigureList`.
- **`DashboardBuilder.build()` heatmap fix:** The similarity heatmap section now properly reconstructs the 2D matrix from the 1D `pairwise_similarities` list using `_reconstruct_similarity_matrix()`. Previously it passed the flat list directly to `similarity_heatmap`, which expected a 2D matrix and could raise `TypeError` or render a broken figure.
- **`HtmlExporter`:** Charts are now laid out in a responsive `auto-fill` CSS Grid container (minimum 480px per column) instead of a fixed 2-column layout, accommodating the additional heatmap chart. KPI cards are rendered as native HTML/CSS elements (colored left bar, hover shadow). The Analysis & Recommendations section uses styled severity badges, grouped card layout, and a summary banner — no longer a plain Plotly annotation.
- **`ReportExporter` base class:** `to_string(report)` and `export(report, path)` now accept an optional `interpretation` keyword argument, enabling rich storytelling in HTML export.
- **Jupyter notebooks** (`pipeline.ipynb`, `pipeline_with_real_embeddings.ipynb`): Updated `DashboardBuilder` usage to chain `.show()` directly on `build()` instead of iterating manually.

### Removed
- **`evaluate_projection` / `ProjectionReport`** and **`evaluate_retrieval` / `RetrievalReport`** (dead code; `projection_quality` and the report's retrieval section cover these).
- **`cosine_similarity_matrix_stats`** — replaced by `pairwise_cosine_stats`.
- **`embedding_scatter_3d`** (redundant with the 2D scatter).
- **`NlpieConfig`** (dead code; thresholds moved to `nlpie/thresholds.py`).
- **`python/nlpie/__init__.pyi`** (mypy resolves re-exports via `__all__`; the compiled-extension stub `_nlpie_core.pyi` remains and is shipped in the wheel).

### Fixed
- `main.py` quick-start runs end-to-end against the current API.
- `DashboardBuilder` without a report raises a clear `ValueError`.
- Effective rank of a degenerate (zero-variance) spectrum returns 0.0 instead of NaN.
- HTML export escapes user-controlled strings (e.g. model names).

---

## [0.1.0] - 2026-07-05

### Added
- Core pipeline with embedding generation.
- Rust–Python interoperability using `pyo3` and `rust-numpy`.
- High-level Python API (`nlpie._api.EmbeddingPreprocessor`).
- Core vector normalization and ops in Rust (PCA whitening, L2 normalization, mean centering, and standardizing columns).
- Initial structure for embedding metrics and operations.

---
