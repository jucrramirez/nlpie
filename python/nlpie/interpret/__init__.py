from .base import Explanation, ExplanationProvider, InterpretationReport
from .hubness import HubnessExplanation, HubInfo, explain_hubness, HubnessExplanationProvider
from .intrinsic import IntrinsicExplanationProvider
from .clustering import ClusteringExplanationProvider
from .geometry import GeometryExplanationProvider
from .projection import ProjectionExplanationProvider
from .retrieval import RetrievalExplanationProvider
from .registry import ExplanationRegistry

ExplanationRegistry.register(HubnessExplanationProvider())
ExplanationRegistry.register(IntrinsicExplanationProvider())
ExplanationRegistry.register(ClusteringExplanationProvider())
ExplanationRegistry.register(GeometryExplanationProvider())
ExplanationRegistry.register(ProjectionExplanationProvider())
ExplanationRegistry.register(RetrievalExplanationProvider())

__all__ = [
    "Explanation",
    "ExplanationProvider",
    "InterpretationReport",
    "HubnessExplanation",
    "HubInfo",
    "explain_hubness",
    "ExplanationRegistry",
]
