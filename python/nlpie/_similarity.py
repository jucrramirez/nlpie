"""Helpers to work with the compact pairwise-similarity representation.

Reports store pairwise cosine similarities as the flattened upper triangle
(``n * (n - 1) / 2`` values). Heatmap renderers reconstruct the full square
matrix on demand through these helpers.
"""

from __future__ import annotations


def reconstruct_similarity_matrix(pairwise: list[float], n: int) -> list[list[float]]:
    """Rebuild the symmetric N x N similarity matrix from its upper triangle.

    ``pairwise`` must contain exactly ``n * (n - 1) / 2`` values in row-major
    order (as produced by :func:`nlpie.pairwise_cosine_stats`).
    """
    expected = n * (n - 1) // 2
    if len(pairwise) != expected:
        raise ValueError(
            f"Cannot rebuild a {n}x{n} similarity matrix from {len(pairwise)} "
            f"values; expected exactly {expected} (the flattened upper triangle)."
        )
    idx = 0
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            val = pairwise[idx]
            matrix[i][j] = val
            matrix[j][i] = val
            idx += 1
    return matrix


def downsample_square_matrix(matrix: list[list[float]], max_size: int) -> list[list[float]]:
    """Uniformly subsample a square matrix to at most ``max_size`` per side."""
    n = len(matrix)
    if n <= max_size:
        return matrix
    step = n // max_size
    indices = list(range(0, n, step))[:max_size]
    return [[matrix[i][j] for j in indices] for i in indices]
