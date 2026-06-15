from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    from . import _shockfinder
except ImportError as exc:  # pragma: no cover - exercised before extension build
    _shockfinder = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(slots=True)
class ShockResult:
    """One row per retained AMR cell."""

    mach: np.ndarray
    shock: np.ndarray
    center_index: np.ndarray
    upstream_index: np.ndarray
    downstream_index: np.ndarray
    selected_indices: np.ndarray


class ShockFinder:
    """Fortran-backed AMR shock finder based on Skillman et al. 2008."""

    def __init__(self) -> None:
        self.maxlevel = 20
        self.minlevel = 0
        self.gamma = 5.0 / 3.0
        self.temperature_floor = 1.0e4
        self.min_mach = 1.0
        self.max_steps = 4
        self.position_unit = "km"
        self.velocity_unit = "km/s"
        self.temperature_unit = "K"
        self.density_unit = "Msol/kpc3"

    def __call__(self, cell: Any) -> ShockResult:
        return self.find(cell)

    def ShockFinder(self, cell: Any) -> ShockResult:
        """Compatibility alias matching the requested example style."""
        return self.find(cell)

    def find(self, cell: Any) -> ShockResult:
        if _shockfinder is None:
            raise ImportError(
                "shocktest Fortran extension is not built. Run "
                "`cd shocktest && python3 -m numpy.f2py -c fortran/shockfinder.f90 "
                "-m _shockfinder` from the project root."
            ) from _IMPORT_ERROR

        arrays = self._extract_amr_arrays(cell)
        selected_indices = arrays.pop("selected_indices")
        neighbors = self._build_neighbors(arrays["pos"], arrays["dx"], arrays["level"])

        n = arrays["temp"].size
        if n == 0:
            empty_float = np.empty(0, dtype=np.float64)
            empty_bool = np.empty(0, dtype=bool)
            empty_index = np.empty(0, dtype=np.int64)
            return ShockResult(
                mach=empty_float,
                shock=empty_bool,
                center_index=empty_index,
                upstream_index=empty_index.copy(),
                downstream_index=empty_index.copy(),
                selected_indices=selected_indices,
            )

        mach, shock, center, upstream, downstream = _shockfinder.shockfinder_kernel.find_shocks(
            arrays["pos"],
            arrays["vel"],
            arrays["dx"],
            arrays["temp"],
            arrays["rho"],
            arrays["level"],
            neighbors,
            float(self.gamma),
            float(self.temperature_floor),
            float(self.min_mach),
            int(self.max_steps),
            n,
        )

        return ShockResult(
            mach=np.asarray(mach, dtype=np.float64),
            shock=np.asarray(shock, dtype=np.int32).astype(bool),
            center_index=self._to_python_indices(center),
            upstream_index=self._to_python_indices(upstream),
            downstream_index=self._to_python_indices(downstream),
            selected_indices=selected_indices,
        )

    def _extract_amr_arrays(self, cell: Any) -> dict[str, np.ndarray]:
        x = self._field(cell, (("x", self.position_unit), "x"))
        y = self._field(cell, (("y", self.position_unit), "y"))
        z = self._field(cell, (("z", self.position_unit), "z"))
        dx = self._field(cell, (("dx", self.position_unit), "dx"))
        vx = self._field(cell, (("vx", self.velocity_unit), "vx"))
        vy = self._field(cell, (("vy", self.velocity_unit), "vy"))
        vz = self._field(cell, (("vz", self.velocity_unit), "vz"))
        temp = self._field(cell, (("T", self.temperature_unit), ("temperature", self.temperature_unit), "T", "temp", "temperature"))
        rho = self._field(cell, (("rho", self.density_unit), "rho", "density"))
        level = self._field(cell, ("level", "levels", "refinement_level")).astype(np.int32, copy=False)

        n = x.size
        fields = {
            "y": y,
            "z": z,
            "dx": dx,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "T": temp,
            "rho": rho,
            "level": level,
        }
        for name, values in fields.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be a 1D AMR cell field")
            if values.size != n:
                raise ValueError(f"{name} has length {values.size}, expected {n}")

        mask = (level >= int(self.minlevel)) & (level <= int(self.maxlevel))
        selected_indices = np.nonzero(mask)[0].astype(np.int64)

        pos = np.asfortranarray(np.column_stack((x[mask], y[mask], z[mask])), dtype=np.float64)
        vel = np.asfortranarray(np.column_stack((vx[mask], vy[mask], vz[mask])), dtype=np.float64)
        return {
            "pos": pos,
            "vel": vel,
            "dx": np.asfortranarray(dx[mask], dtype=np.float64),
            "temp": np.asfortranarray(temp[mask], dtype=np.float64),
            "rho": np.asfortranarray(rho[mask], dtype=np.float64),
            "level": np.asfortranarray(level[mask], dtype=np.int32),
            "selected_indices": selected_indices,
        }

    @staticmethod
    def _to_python_indices(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.int64) - 1
        out[out < 0] = -1
        return out

    @staticmethod
    def _field(cell: Any, names: tuple[Any, ...]) -> np.ndarray:
        for name in names:
            try:
                value = ShockFinder._get(cell, name)
            except (KeyError, TypeError, IndexError):
                continue
            if value is not None:
                arr = np.asarray(value)
                if arr.ndim != 1:
                    raise ValueError(f"{name!r} must be a 1D AMR cell field")
                return arr
        wanted = ", ".join(repr(name) for name in names)
        raise KeyError(f"cell is missing required field; tried {wanted}")

    @staticmethod
    def _get(cell: Any, name: Any) -> Any:
        if isinstance(cell, Mapping):
            if name in cell:
                return cell[name]
            raise KeyError(name)
        try:
            return cell[name]
        except (KeyError, TypeError, IndexError):
            if isinstance(name, str) and hasattr(cell, name):
                return getattr(cell, name)
            raise

    @staticmethod
    def _build_neighbors(pos: np.ndarray, dx: np.ndarray, level: np.ndarray) -> np.ndarray:
        n = dx.size
        neighbors = np.zeros((n, 6), dtype=np.int32, order="F")
        if n == 0:
            return neighbors

        if np.any(dx <= 0.0):
            raise ValueError("dx must be positive for every retained cell")

        finest_dx = float(np.min(dx))
        origin = np.min(pos - dx[:, None] * 0.5, axis=0)
        lo = ShockFinder._quantize((pos - dx[:, None] * 0.5 - origin) / finest_dx)
        hi = ShockFinder._quantize((pos + dx[:, None] * 0.5 - origin) / finest_dx)
        widths = hi - lo

        if np.any(widths <= 0):
            raise ValueError("cell boxes collapsed during AMR quantization")
        if np.any(np.max(widths, axis=1) != np.min(widths, axis=1)):
            raise ValueError("AMR cells must be cubic")

        same_level: dict[tuple[int, int, int, int], int] = {}
        boxes_by_width: dict[int, dict[tuple[int, int, int], int]] = {}
        for idx in range(n):
            key = (int(level[idx]), int(lo[idx, 0]), int(lo[idx, 1]), int(lo[idx, 2]))
            same_level[key] = idx
            width = int(widths[idx, 0])
            box_key = (int(lo[idx, 0]), int(lo[idx, 1]), int(lo[idx, 2]))
            boxes_by_width.setdefault(width, {})[box_key] = idx

        coarser_widths = sorted(boxes_by_width, reverse=True)
        for idx in range(n):
            width = int(widths[idx, 0])
            center0 = 0.5 * (lo[idx, 0] + hi[idx, 0])
            center1 = 0.5 * (lo[idx, 1] + hi[idx, 1])
            center2 = 0.5 * (lo[idx, 2] + hi[idx, 2])
            for axis in range(3):
                for direction, face_offset in ((-1, 0), (1, 1)):
                    face = axis * 2 + face_offset
                    shifted0 = int(lo[idx, 0])
                    shifted1 = int(lo[idx, 1])
                    shifted2 = int(lo[idx, 2])
                    if axis == 0:
                        shifted0 += direction * width
                    elif axis == 1:
                        shifted1 += direction * width
                    else:
                        shifted2 += direction * width
                    key = (
                        int(level[idx]),
                        shifted0,
                        shifted1,
                        shifted2,
                    )
                    same = same_level.get(key)
                    if same is not None and np.all(widths[same] == widths[idx]):
                        neighbors[idx, face] = same + 1
                        continue

                    sample0 = center0
                    sample1 = center1
                    sample2 = center2
                    if axis == 0:
                        sample0 += direction * (width * 0.5 + 0.25)
                    elif axis == 1:
                        sample1 += direction * (width * 0.5 + 0.25)
                    else:
                        sample2 += direction * (width * 0.5 + 0.25)
                    lower = ShockFinder._find_lower_level_neighbor(
                        sample0,
                        sample1,
                        sample2,
                        width,
                        coarser_widths,
                        boxes_by_width,
                    )
                    if lower >= 0:
                        neighbors[idx, face] = lower + 1

        return neighbors

    @staticmethod
    def _find_lower_level_neighbor(
        sample0: float,
        sample1: float,
        sample2: float,
        current_width: int,
        coarser_widths: list[int],
        boxes_by_width: dict[int, dict[tuple[int, int, int], int]],
    ) -> int:
        for width in coarser_widths:
            if width <= current_width:
                continue
            key = (
                int(np.floor(sample0 / width)) * width,
                int(np.floor(sample1 / width)) * width,
                int(np.floor(sample2 / width)) * width,
            )
            idx = boxes_by_width[width].get(key)
            if idx is not None:
                return idx
        return -1

    @staticmethod
    def _quantize(values: np.ndarray) -> np.ndarray:
        return np.floor(values + 0.5).astype(np.int64)
