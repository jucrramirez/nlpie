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
- Fixed compiler warnings by removing unused imports/variables.
- Updated documentation comments for new modules and functions.
- Updated Python bindings to expose the new metrics.

---

## [0.1.0] - 2026-07-05

### Added
- Core pipeline with embedding generation.
- Rust–Python interoperability using `pyo3` and `rust-numpy`.
- High-level Python API (`nlpie._api.EmbeddingPreprocessor`).
- Core vector normalization and ops in Rust (PCA whitening, L2 normalization, mean centering, and standardizing columns).
- Initial structure for embedding metrics and operations.

---
