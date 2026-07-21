from __future__ import annotations

from .base import Explanation, ExplanationProvider, InterpretationReport


class ExplanationRegistry:
    _providers: dict[str, ExplanationProvider] = {}

    @classmethod
    def register(cls, provider: ExplanationProvider) -> None:
        for key in provider.metric_keys():
            cls._providers[key] = provider

    @classmethod
    def get_provider(cls, metric_key: str) -> ExplanationProvider | None:
        return cls._providers.get(metric_key)

    @classmethod
    def explain(cls, metric_key: str, report) -> list[Explanation]:
        provider = cls.get_provider(metric_key)
        if provider is not None:
            return provider.explain(report)
        return []

    @classmethod
    def explain_all(cls, report) -> InterpretationReport:
        seen: set[int] = set()
        explanations: list[Explanation] = []
        for provider in cls._providers.values():
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            explanations.extend(provider.explain(report))
        return InterpretationReport(explanations=explanations)
