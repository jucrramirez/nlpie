from .base import Explanation, ExplanationProvider, InterpretationReport
from .clustering import ClusteringExplanationProvider
from .geometry import GeometryExplanationProvider
from .hubness import HubInfo, HubnessExplanation, HubnessExplanationProvider, explain_hubness
from .intrinsic import IntrinsicExplanationProvider
from .projection import ProjectionExplanationProvider
from .registry import ExplanationRegistry
from .retrieval import RetrievalExplanationProvider
from .story import StoryData, StoryGroup, build_story

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
    "HubnessExplanationProvider",
    "IntrinsicExplanationProvider",
    "ClusteringExplanationProvider",
    "GeometryExplanationProvider",
    "ProjectionExplanationProvider",
    "RetrievalExplanationProvider",
    "StoryData",
    "StoryGroup",
    "build_story",
]
