from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Explanation:
    metric: str
    severity: str
    summary: str
    detail: str
    recommendation: str


@dataclass
class InterpretationReport:
    explanations: list[Explanation] = field(default_factory=list)

    def by_severity(self, severity: str) -> list[Explanation]:
        return [e for e in self.explanations if e.severity == severity]

    @property
    def critical(self) -> list[Explanation]:
        return self.by_severity("critical")

    @property
    def warnings(self) -> list[Explanation]:
        return self.by_severity("warning")

    @property
    def info(self) -> list[Explanation]:
        return self.by_severity("info")

    def __bool__(self) -> bool:
        return len(self.explanations) > 0

    def __str__(self) -> str:
        if not self.explanations:
            return "Interpretation Report: No issues detected."
        lines = ["Interpretation Report", "=" * 60]
        for sev in ("critical", "warning", "info"):
            items = self.by_severity(sev)
            if items:
                lines.append(f"\n{sev.upper()}:")
                for e in items:
                    lines.append(f"  [{e.metric}] {e.summary}")
        return "\n".join(lines)


class ExplanationProvider(ABC):
    @abstractmethod
    def metric_keys(self) -> list[str]:
        ...

    @abstractmethod
    def explain(self, report) -> Optional[Explanation]:
        ...
