from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

# A 2D matrix-like structure (e.g. list of lists, tuple of tuples, or numpy array)
MatrixLike = Sequence[Sequence[float]] | np.ndarray[Any, Any]

# A 1D vector-like structure (e.g. list of floats, tuple of floats, or numpy array)
VectorLike = Sequence[float] | np.ndarray[Any, Any]

DistanceMetric = Literal["cosine", "euclidean"]
