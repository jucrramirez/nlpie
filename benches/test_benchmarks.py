"""Performance benchmarks for the hot Rust paths.

Run explicitly (not part of the default suite):

    uv run pytest benches/ --benchmark-only

Compare against a saved baseline:

    uv run pytest benches/ --benchmark-only --benchmark-save=base
    uv run pytest benches/ --benchmark-only --benchmark-compare=base
"""

import nlpie
import numpy as np
import pytest

N_SAMPLES = 500
N_DIMS = 128


@pytest.fixture(scope="module")
def embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_SAMPLES, N_DIMS)).astype(np.float32)


@pytest.fixture(scope="module")
def labels() -> list[int]:
    return [i % 10 for i in range(N_SAMPLES)]


def test_pairwise_cosine_stats(benchmark, embeddings):
    benchmark(nlpie.pairwise_cosine_stats, embeddings)


def test_projection_quality(benchmark, embeddings):
    rng = np.random.default_rng(1)
    low = rng.standard_normal((N_SAMPLES, 2)).astype(np.float32)
    benchmark(nlpie.projection_quality, embeddings, low, [5, 10, 20])


def test_silhouette_score(benchmark, embeddings, labels):
    benchmark(nlpie.silhouette_score, embeddings, labels)


def test_compute_hubness(benchmark, embeddings):
    benchmark(nlpie.compute_hubness, embeddings, 5)


def test_evaluate_embedding_quality(benchmark, embeddings, labels):
    benchmark(nlpie.evaluate_embedding_quality, embeddings, labels=labels, hubness_k=5)
