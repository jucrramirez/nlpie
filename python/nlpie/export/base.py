from __future__ import annotations

from abc import ABC, abstractmethod


class ReportExporter(ABC):
    @abstractmethod
    def export(self, report, path: str, interpretation=None) -> None: ...

    @abstractmethod
    def to_string(self, report, interpretation=None) -> str: ...
