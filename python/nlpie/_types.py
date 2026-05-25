"""Type definitions and protocols for the nlpie package."""

from typing import Sequence, Union

# A 2D matrix-like structure (e.g. list of lists, tuple of tuples, or numpy array)
MatrixLike = Sequence[Sequence[float]]

# A 1D vector-like structure (e.g. list of floats, tuple of floats, or numpy array)
VectorLike = Sequence[float]
