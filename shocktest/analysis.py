from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import ShockResult


@dataclass(slots=True)
class ShockAnalysis:
    """Products and timings from a single-pass shock analysis."""

    result: ShockResult | None
    dissipation: Any | None
    catalog: Any | None
    timings: dict[str, float]

    @property
    def counts(self) -> dict[str, int]:
        result = self.result
        if result is None:
            return {"retained": 0, "shock": 0, "representative": 0, "groups": 0}
        shock_count = int(result.shock.sum())
        representative_count = shock_count
        group_count = 0
        if self.catalog is not None:
            representatives = self.catalog.center_representative
            representative_count = int(
                np.count_nonzero(
                    (representatives >= 0)
                    & (representatives == np.arange(result.mach.size))
                )
            )
            group_count = len(self.catalog.groups)
        return {
            "retained": int(result.mach.size),
            "shock": shock_count,
            "representative": representative_count,
            "groups": group_count,
        }

    def clear(self) -> None:
        if self.result is not None:
            self.result.clear()
            self.result = None
        if self.dissipation is not None:
            self.dissipation.clear()
            self.dissipation = None
        self.catalog = None
        self.timings.clear()
        gc.collect()
