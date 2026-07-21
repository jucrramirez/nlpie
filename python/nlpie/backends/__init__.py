from .base import Dashboard, PlotBackend
from .plotly import PlotlyBackend


def _resolve_backend(backend: PlotBackend | None = None) -> PlotBackend:
    """Return the given backend, defaulting to Plotly when ``None``."""
    if backend is None:
        return PlotlyBackend()
    return backend


__all__ = [
    "Dashboard",
    "PlotBackend",
    "PlotlyBackend",
]
