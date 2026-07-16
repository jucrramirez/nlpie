"""NLPie — Quick-start entry point.

Usage:
    uv run python main.py
"""

import numpy as np

from nlpie import (
    EmbeddingPreprocessor,
    evaluate_embedding_quality,
    plot_quality_report,
)


def main():
    np.random.seed(42)
    n, d, n_clusters = 200, 64, 5

    embeddings = np.random.randn(n, d).astype(np.float32)
    labels = [i % n_clusters for i in range(n)]

    preprocessor = EmbeddingPreprocessor()
    normalized = preprocessor.l2_normalize_rows(embeddings)

    report, interpretation = evaluate_embedding_quality(
        normalized,
        labels=labels,
        hubness_k=5,
        model_name="example",
    )

    print(report)
    print()
    print(interpretation)

    try:
        fig = plot_quality_report(report)
        fig.show()
    except ImportError:
        print("Install plotly for interactive charts: uv sync --extra plotting")


if __name__ == "__main__":
    main()
