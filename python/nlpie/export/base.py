from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ReportExporter(ABC):
    @abstractmethod
    def export(self, report, path: str) -> None:
        ...

    @abstractmethod
    def to_string(self, report, interpretation=None) -> str:
        ...
