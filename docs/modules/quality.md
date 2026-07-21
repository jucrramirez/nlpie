# Quality Report Module API Reference

The `metrics.quality` module provides a high-level wrapper to evaluate the overall quality of an embedding space across multiple axes simultaneously.

## `evaluate_embedding_quality`

Generates an `EmbeddingQualityReport` by computing intrinsic, geometry, clustering, projection, and retrieval metrics, plus an `InterpretationReport` with plain-English explanations. The function returns a `(report, interpretation)` tuple. The report only computes the metrics for which the necessary inputs are provided, and all inputs are validated up front (invalid shapes or out-of-range `k` values raise immediately instead of being silently clamped).

```python
from nlpie import evaluate_embedding_quality

report, interpretation = evaluate_embedding_quality(
    embeddings=my_embeddings_matrix,

    # Optional: for clustering metrics
    labels=ground_truth_labels,
    labels_pred=predicted_labels,

    # Optional: for projection metrics
    low_dim=umap_embeddings_matrix,
    projection_k_values=[5, 10, 20],

    # Optional: for retrieval metrics
    retrieved=retrieved_lists,
    relevant=relevant_lists,
    retrieval_k_values=[1, 5, 10],

    # Diagnostics
    hubness_k=5,
    model_name="baseline-v1"
)

print(report)
print(interpretation)
```

> **Note on `k` bounds:** projection neighbourhood sizes must satisfy
> `1 <= k < n` and `2n - 3k - 1 > 0` (the metric's denominator must be
> positive). With the default `projection_k_values=(5, 10, 20)` this means
> projections need at least 31 samples; pass smaller `k` values (or omit
> `low_dim`) for tiny datasets.

## `compare_models`

Evaluates multiple models side-by-side using the same labels, relevant documents, etc.

```python
from nlpie import compare_models

reports = compare_models(
    models={
        "baseline": base_embeddings,
        "finetuned": finetuned_embeddings
    },
    labels=shared_labels,
    retrieved={
        "baseline": base_retrieved,
        "finetuned": finetuned_retrieved
    },
    relevant=shared_relevant
)

for report in reports:
    print(report)
```

## The `EmbeddingQualityReport` Data Structure

The report object contains dataclasses for each category of metrics. These can be accessed programmatically.

- **`report.intrinsic`**: `mean`, `std`, `min`, `max` cosine similarity.
- **`report.clustering`**: `ari`, `nmi`, `purity`, `silhouette`, `calinski_harabasz`.
- **`report.geometry`**: `effective_rank`, `mean_similarity` (to centroid), `hubness_skewness`, `hubness_k`.
- **`report.projection`**: List of objects with `k`, `trustworthiness`, `continuity`.
- **`report.retrieval`**: List of objects with `k`, `recall`, `precision`, `mrr`, `ndcg`, `coverage`.

All sections that were skipped due to missing inputs will be `None` or empty lists.

## The `InterpretationReport` Data Structure

Each computed metric section produces one or more `Explanation` objects with:

- **`metric`**: which metric family it describes (e.g. `"hubness"`, `"projection"`).
- **`severity`**: `"critical"`, `"warning"`, or `"info"` (breakpoints live in `nlpie/thresholds.py`).
- **`summary`**: one-line plain-English status.
- **`detail`**: what the value means for downstream tasks.
- **`recommendation`**: actionable remediation step.

Severity thresholds are shared with the dashboard KPI cards, so a metric shown as red in a chart is explained with the same severity in the report.
