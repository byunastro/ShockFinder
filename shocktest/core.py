from __future__ import annotations

import gc
import time
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
    pos: np.ndarray | None = None
    dx: np.ndarray | None = None
    normal: np.ndarray | None = None
    level: np.ndarray | None = None
    zone_width: np.ndarray | None = None

    def clear(self) -> None:
        """Release arrays held by this result object."""

        empty_float = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=bool)
        empty_index = np.empty(0, dtype=np.int64)
        self.mach = empty_float
        self.shock = empty_bool
        self.center_index = empty_index
        self.upstream_index = empty_index.copy()
        self.downstream_index = empty_index.copy()
        self.selected_indices = empty_index.copy()
        self.pos = None
        self.dx = None
        self.normal = None
        self.level = None
        self.zone_width = None
        gc.collect()


class ShockFinder:
    """Fortran-backed AMR shock finder based on Skillman et al. 2008."""

    def __init__(self) -> None:
        self.maxlevel = 20
        self.minlevel = 0
        self.gamma = 5.0 / 3.0
        self.temperature_floor = 1.0e4
        self.min_temperature = None
        self.min_density = None
        self.max_density = None
        self.min_mach = 1.0
        self.max_steps = 4
        self.position_unit = "km"
        self.velocity_unit = "km/s"
        self.temperature_unit = "K"
        self.density_unit = "Msol/kpc3"
        self.show_progress = False
        self.progress_interval = 0
        self.boundary = "open"

    def __call__(self, cell: Any) -> ShockResult:
        return self.find(cell)

    def ShockFinder(self, cell: Any) -> ShockResult:
        """Compatibility alias matching the requested example style."""
        return self.find(cell)

    def analyze(
        self,
        cell: Any,
        *,
        compute_dissipation: bool = True,
        build_catalog: bool = True,
        deduplicate: bool = True,
        mach_tolerance: float = 0.3,
        normal_cosine: float = 0.7,
        duplicate_normal_cosine: float = 0.8,
        min_mach: float | None = None,
        external_temperature: float = 1.0e4,
        classification_fraction: float = 0.8,
        dissipation_options: dict[str, Any] | None = None,
    ):
        """Run shock detection and optional post-processing in one pass.

        The AMR neighbor tables are retained only until catalog construction is
        complete, avoiding the second geometry build required by the separate
        ``find`` then ``build_shock_catalog`` workflow.
        """

        from .analysis import ShockAnalysis
        from .catalog import build_shock_catalog as make_catalog
        from .pyShockFinder import compute_dissipation as make_dissipation

        analysis_start = time.perf_counter()
        result, neighbor_tables, timings = self._find_internal(cell)
        dissipation = None
        catalog = None
        if compute_dissipation:
            stage_start = time.perf_counter()
            options = {} if dissipation_options is None else dict(dissipation_options)
            if "gamma" in options and not np.isclose(float(options["gamma"]), float(self.gamma)):
                raise ValueError(
                    "dissipation gamma must match finder.gamma for a physically "
                    "consistent single-pass analysis"
                )
            options["gamma"] = float(self.gamma)
            dissipation = make_dissipation(
                cell, result, **options
            )
            timings["dissipation"] = time.perf_counter() - stage_start
        if build_catalog:
            stage_start = time.perf_counter()
            catalog = make_catalog(
                result,
                cell=cell,
                dissipation=dissipation,
                mach_tolerance=mach_tolerance,
                normal_cosine=normal_cosine,
                deduplicate=deduplicate,
                duplicate_normal_cosine=duplicate_normal_cosine,
                min_mach=self.min_mach if min_mach is None else min_mach,
                boundary=self.boundary,
                external_temperature=external_temperature,
                classification_fraction=classification_fraction,
                _neighbor_tables=neighbor_tables,
            )
            timings["catalog"] = time.perf_counter() - stage_start
        timings["total"] = time.perf_counter() - analysis_start
        return ShockAnalysis(
            result=result,
            dissipation=dissipation,
            catalog=catalog,
            timings=timings,
        )

    def clear(self) -> None:
        """Compatibility cleanup hook.

        ``ShockFinder`` does not keep large AMR arrays after each run; those are
        owned by the returned ``ShockResult``. Call ``result.clear()`` to release
        result arrays explicitly.
        """

        gc.collect()

    def find(self, cell: Any) -> ShockResult:
        result, _, _ = self._find_internal(cell)
        return result

    def _find_internal(self, cell: Any):
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        self._validate_settings()
        if self.boundary != "open":
            raise ValueError(
                "boundary must be 'open'; ShockFinder inputs are extracted "
                "snapshot regions and their outer faces are not connected"
            )
        if _shockfinder is None:
            raise ImportError(
                "shocktest Fortran extension is not built. Run "
                "`cd shocktest && python3 -m numpy.f2py -c fortran/shockfinder.f90 "
                "-m _shockfinder` from the project root."
            ) from _IMPORT_ERROR

        self._progress("ShockFinder: reading AMR cell fields")
        stage_start = time.perf_counter()
        arrays = self._extract_amr_arrays(cell)
        timings["input"] = time.perf_counter() - stage_start
        selected_indices = arrays.pop("selected_indices")
        n = arrays["temp"].size
        self._progress(f"ShockFinder: retained {n:,} cells after filtering")

        interval = self._resolved_progress_interval(n)
        self._progress("ShockFinder: building AMR face-neighbor table")
        stage_start = time.perf_counter()
        neighbors, fine_neighbors = self._build_neighbor_tables(
            arrays["pos"],
            arrays["dx"],
            arrays["level"],
            show_progress=self.show_progress,
            progress_interval=interval,
        )
        timings["neighbors"] = time.perf_counter() - stage_start

        if n == 0:
            empty_float = np.empty(0, dtype=np.float64)
            empty_bool = np.empty(0, dtype=bool)
            empty_index = np.empty(0, dtype=np.int64)
            result = ShockResult(
                mach=empty_float,
                shock=empty_bool,
                center_index=empty_index,
                upstream_index=empty_index.copy(),
                downstream_index=empty_index.copy(),
                selected_indices=selected_indices,
                pos=np.empty((0, 3), dtype=np.float64),
                dx=empty_float.copy(),
                normal=np.empty((0, 3), dtype=np.float64),
                level=np.empty(0, dtype=np.int32),
                zone_width=empty_float.copy(),
            )
            timings["scan"] = 0.0
            timings["detection_total"] = time.perf_counter() - total_start
            return result, (neighbors, fine_neighbors), timings

        self._progress("ShockFinder: running Fortran shock scan")
        stage_start = time.perf_counter()
        mach, shock, center, upstream, downstream = _shockfinder.shockfinder_kernel.find_shocks(
            arrays["pos"],
            arrays["vel"],
            arrays["dx"],
            arrays["temp"],
            arrays["rho"],
            arrays["level"],
            neighbors,
            fine_neighbors,
            float(self.gamma),
            float(self.temperature_floor),
            float(self.min_mach),
            int(self.max_steps),
            int(bool(self.show_progress)),
            int(interval),
            n,
        )
        timings["scan"] = time.perf_counter() - stage_start
        self._progress("ShockFinder: done")

        result_pos = arrays["pos"]
        result_dx = arrays["dx"]
        result_level = arrays["level"]
        normal = np.zeros_like(result_pos)
        zone_width = np.zeros(n, dtype=np.float64)
        valid_normal = (upstream > 0) & (downstream > 0)
        if np.any(valid_normal):
            upstream_rows = np.asarray(upstream[valid_normal], dtype=np.int64) - 1
            downstream_rows = np.asarray(downstream[valid_normal], dtype=np.int64) - 1
            vectors = result_pos[downstream_rows] - result_pos[upstream_rows]
            lengths = np.linalg.norm(vectors, axis=1)
            zone_width[np.nonzero(valid_normal)[0]] = lengths
            nonzero = lengths > 0.0
            valid_rows = np.nonzero(valid_normal)[0]
            normal[valid_rows[nonzero]] = vectors[nonzero] / lengths[nonzero, None]
        del arrays

        result = ShockResult(
            mach=np.asarray(mach, dtype=np.float64),
            shock=np.asarray(shock, dtype=np.int32).astype(bool),
            center_index=self._to_python_indices(center),
            upstream_index=self._to_python_indices(upstream),
            downstream_index=self._to_python_indices(downstream),
            selected_indices=selected_indices,
            pos=result_pos,
            dx=result_dx,
            normal=normal,
            level=result_level,
            zone_width=zone_width,
        )
        timings["detection_total"] = time.perf_counter() - total_start
        return result, (neighbors, fine_neighbors), timings

    def _validate_settings(self) -> None:
        if not np.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than 1")
        if not np.isfinite(self.temperature_floor) or self.temperature_floor <= 0.0:
            raise ValueError("temperature_floor must be finite and positive")
        if not np.isfinite(self.min_mach) or self.min_mach < 1.0:
            raise ValueError("min_mach must be finite and at least 1")
        if (
            isinstance(self.max_steps, (bool, np.bool_))
            or int(self.max_steps) != self.max_steps
            or self.max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer")
        if int(self.minlevel) > int(self.maxlevel):
            raise ValueError("minlevel must not exceed maxlevel")

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
        level_values = self._field(cell, ("level", "levels", "refinement_level"))
        if not np.all(np.isfinite(level_values)):
            raise ValueError("level must contain only finite values")
        if not np.all(level_values == np.floor(level_values)):
            raise ValueError("level must contain integer values")
        level = level_values.astype(np.int32, copy=False)

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
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")

        # Keep only cells requested by the caller. Geometry still comes from dx,
        # not from the level filter.
        mask = (level >= int(self.minlevel)) & (level <= int(self.maxlevel))
        if self.min_temperature is not None:
            mask &= temp >= float(self.min_temperature)
        if self.min_density is not None:
            mask &= rho >= float(self.min_density)
        if self.max_density is not None:
            mask &= rho <= float(self.max_density)
        selected_indices = np.nonzero(mask)[0].astype(np.int64, copy=False)
        del mask

        # f2py passes Fortran-contiguous arrays to the compiled kernel without
        # needing extra copies. Fill the columns directly to avoid the
        # additional dense temporary made by column_stack.
        n_selected = selected_indices.size
        pos = np.empty((n_selected, 3), dtype=np.float64, order="F")
        vel = np.empty((n_selected, 3), dtype=np.float64, order="F")
        pos[:, 0] = x[selected_indices]
        pos[:, 1] = y[selected_indices]
        pos[:, 2] = z[selected_indices]
        vel[:, 0] = vx[selected_indices]
        vel[:, 1] = vy[selected_indices]
        vel[:, 2] = vz[selected_indices]
        return {
            "pos": pos,
            "vel": vel,
            "dx": np.asfortranarray(dx[selected_indices], dtype=np.float64),
            "temp": np.asfortranarray(temp[selected_indices], dtype=np.float64),
            "rho": np.asfortranarray(rho[selected_indices], dtype=np.float64),
            "level": np.asfortranarray(level[selected_indices], dtype=np.int32),
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
    def _build_neighbors(
        pos: np.ndarray,
        dx: np.ndarray,
        level: np.ndarray,
        *,
        show_progress: bool = False,
        progress_interval: int = 0,
    ) -> np.ndarray:
        neighbors, _ = ShockFinder._build_neighbor_tables(
            pos,
            dx,
            level,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        return neighbors

    @staticmethod
    def _build_neighbor_tables(
        pos: np.ndarray,
        dx: np.ndarray,
        level: np.ndarray,
        *,
        show_progress: bool = False,
        progress_interval: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = dx.size
        neighbors = np.zeros((n, 6), dtype=np.int32, order="F")
        fine_neighbors = np.zeros((n, 6, 4), dtype=np.int32, order="F")
        if n == 0:
            return neighbors, fine_neighbors

        if np.any(dx <= 0.0):
            raise ValueError("dx must be positive for every retained cell")

        finest_dx = float(np.min(dx))
        half_dx = 0.5 * dx
        box_work = np.empty(pos.shape, dtype=np.float64)
        np.subtract(pos, half_dx[:, None], out=box_work)
        origin = np.min(box_work, axis=0)

        # Convert floating-point AMR boxes into integer boxes measured in units
        # of the finest retained cell width. This makes neighbor lookup exact
        # enough for AMR grids while avoiding repeated floating comparisons.
        box_work -= origin
        box_work /= finest_dx
        lo = ShockFinder._quantize_inplace(box_work)

        np.add(pos, half_dx[:, None], out=box_work)
        box_work -= origin
        box_work /= finest_dx
        hi = ShockFinder._quantize_inplace(box_work)
        del box_work, half_dx

        widths = hi - lo
        cell_width = widths[:, 0].copy()

        if np.any(widths <= 0):
            raise ValueError("cell boxes collapsed during AMR quantization")
        if np.any(np.max(widths, axis=1) != np.min(widths, axis=1)):
            raise ValueError("AMR cells must be cubic")
        del hi, widths

        spans = tuple(int(np.max(lo[:, axis] + cell_width)) + 1 for axis in range(3))
        spatial_size = spans[0] * spans[1] * spans[2]
        level_min = int(np.min(level))
        level_count = int(np.max(level)) - level_min + 1
        if spatial_size * max(level_count, 1) > np.iinfo(np.uint64).max:
            raise OverflowError("AMR integer box range is too large for sorted-key neighbor lookup")

        stage_start = time.perf_counter()
        spatial_key = ShockFinder._pack_spatial(lo[:, 0], lo[:, 1], lo[:, 2], spans)
        same_key = spatial_key + (level.astype(np.uint64) - np.uint64(level_min)) * np.uint64(spatial_size)
        same_order = np.argsort(same_key, kind="mergesort")
        same_keys_sorted = same_key[same_order]
        if show_progress:
            ShockFinder._print_progress("ShockFinder: indexing AMR boxes", n, n, stage_start)

        unique_widths = np.unique(cell_width)
        width_lookups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for width in unique_widths:
            rows = np.nonzero(cell_width == width)[0]
            keys = spatial_key[rows]
            order = np.argsort(keys, kind="mergesort")
            width_lookups[int(width)] = (keys[order], rows[order])

        stage_start = time.perf_counter()
        base0 = lo[:, 0]
        base1 = lo[:, 1]
        base2 = lo[:, 2]
        for axis in range(3):
            for direction, face_offset in ((-1, 0), (1, 1)):
                face = axis * 2 + face_offset
                target0 = base0.copy()
                target1 = base1.copy()
                target2 = base2.copy()
                if axis == 0:
                    target0 += direction * cell_width
                elif axis == 1:
                    target1 += direction * cell_width
                else:
                    target2 += direction * cell_width

                valid = (
                    (target0 >= 0)
                    & (target0 < spans[0])
                    & (target1 >= 0)
                    & (target1 < spans[1])
                    & (target2 >= 0)
                    & (target2 < spans[2])
                )
                target_key = np.zeros(n, dtype=np.uint64)
                target_key[valid] = ShockFinder._pack_spatial(target0[valid], target1[valid], target2[valid], spans)
                target_key[valid] += (level[valid].astype(np.uint64) - np.uint64(level_min)) * np.uint64(spatial_size)
                same = ShockFinder._lookup_sorted(target_key, same_keys_sorted, same_order)
                same_valid = valid & (same >= 0) & (cell_width[np.maximum(same, 0)] == cell_width)
                neighbors[same_valid, face] = same[same_valid].astype(np.int32) + 1

                del target0, target1, target2, target_key, same, same_valid, valid
                if show_progress:
                    face_done = axis * 2 + face_offset + 1
                    ShockFinder._print_progress(
                        "ShockFinder: linking same-level AMR neighbors",
                        min(n, (n * face_done) // 6),
                        n,
                        stage_start,
                    )

        coarser_widths = sorted((int(width) for width in unique_widths), reverse=True)
        for width in unique_widths:
            rows_for_width = np.nonzero(cell_width == width)[0]
            width_i = int(width)
            center0 = lo[rows_for_width, 0] + 0.5 * width_i
            center1 = lo[rows_for_width, 1] + 0.5 * width_i
            center2 = lo[rows_for_width, 2] + 0.5 * width_i
            for axis in range(3):
                for direction, face_offset in ((-1, 0), (1, 1)):
                    face = axis * 2 + face_offset
                    missing = rows_for_width[neighbors[rows_for_width, face] == 0]
                    if missing.size == 0:
                        continue
                    sample0 = center0[neighbors[rows_for_width, face] == 0].copy()
                    sample1 = center1[neighbors[rows_for_width, face] == 0].copy()
                    sample2 = center2[neighbors[rows_for_width, face] == 0].copy()
                    if axis == 0:
                        sample0 += direction * (width_i * 0.5 + 0.25)
                    elif axis == 1:
                        sample1 += direction * (width_i * 0.5 + 0.25)
                    else:
                        sample2 += direction * (width_i * 0.5 + 0.25)
                    for coarse_width in coarser_widths:
                        if coarse_width <= width_i:
                            continue
                        keys_sorted, indices_sorted = width_lookups[coarse_width]
                        target0 = np.floor(sample0 / coarse_width).astype(np.int64) * coarse_width
                        target1 = np.floor(sample1 / coarse_width).astype(np.int64) * coarse_width
                        target2 = np.floor(sample2 / coarse_width).astype(np.int64) * coarse_width
                        valid = (
                            (target0 >= 0)
                            & (target0 < spans[0])
                            & (target1 >= 0)
                            & (target1 < spans[1])
                            & (target2 >= 0)
                            & (target2 < spans[2])
                        )
                        if not np.any(valid):
                            continue
                        target_key = np.zeros(missing.size, dtype=np.uint64)
                        target_key[valid] = ShockFinder._pack_spatial(target0[valid], target1[valid], target2[valid], spans)
                        found = ShockFinder._lookup_sorted(target_key, keys_sorted, indices_sorted)
                        found_valid = valid & (found >= 0)
                        neighbors[missing[found_valid], face] = found[found_valid].astype(np.int32) + 1
                        keep = ~found_valid
                        if not np.any(keep):
                            break
                        missing = missing[keep]
                        sample0 = sample0[keep]
                        sample1 = sample1[keep]
                        sample2 = sample2[keep]
            if show_progress:
                ShockFinder._print_progress(
                    "ShockFinder: linking same-or-coarser AMR neighbors",
                    int(np.searchsorted(unique_widths, width, side="right")),
                    unique_widths.size,
                    stage_start,
                )

        if show_progress:
            ShockFinder._print_progress("ShockFinder: linking AMR neighbors", n, n, stage_start)

        fine_widths = coarser_widths
        fine_stage_start = time.perf_counter()
        for width in unique_widths:
            width_i = int(width)
            finer_width = next((candidate for candidate in fine_widths if candidate < width), None)
            if finer_width is None:
                continue
            ratio = width_i // finer_width
            if ratio <= 0 or width_i % finer_width != 0:
                continue
            if ratio > 2:
                raise ValueError(
                    "fine AMR face neighbors currently support one refinement jump "
                    "(2x per axis, 4 cells per face). Include intermediate AMR "
                    "levels in the retained region or narrow the level range."
                )
            rows_for_width = np.nonzero(cell_width == width)[0]
            keys_sorted, indices_sorted = width_lookups[finer_width]

            for axis in range(3):
                other_axes = [other for other in range(3) if other != axis]
                for direction, face_offset in ((-1, 0), (1, 1)):
                    face = axis * 2 + face_offset
                    slot = 0
                    for offset0 in range(ratio):
                        for offset1 in range(ratio):
                            if slot >= 4:
                                break
                            target0 = lo[rows_for_width, 0].copy()
                            target1 = lo[rows_for_width, 1].copy()
                            target2 = lo[rows_for_width, 2].copy()
                            targets = (target0, target1, target2)
                            targets[axis][:] = lo[rows_for_width, axis] - finer_width if direction < 0 else lo[rows_for_width, axis] + width_i
                            targets[other_axes[0]][:] = lo[rows_for_width, other_axes[0]] + offset0 * finer_width
                            targets[other_axes[1]][:] = lo[rows_for_width, other_axes[1]] + offset1 * finer_width
                            valid = (
                                (target0 >= 0)
                                & (target0 < spans[0])
                                & (target1 >= 0)
                                & (target1 < spans[1])
                                & (target2 >= 0)
                                & (target2 < spans[2])
                            )
                            target_key = np.zeros(rows_for_width.size, dtype=np.uint64)
                            target_key[valid] = ShockFinder._pack_spatial(target0[valid], target1[valid], target2[valid], spans)
                            found = ShockFinder._lookup_sorted(target_key, keys_sorted, indices_sorted)
                            found_valid = valid & (found >= 0)
                            fine_neighbors[rows_for_width[found_valid], face, slot] = found[found_valid].astype(np.int32) + 1
                            slot += 1
            if show_progress:
                ShockFinder._print_progress(
                    "ShockFinder: linking finer AMR face neighbors",
                    int(np.searchsorted(unique_widths, width, side="right")),
                    unique_widths.size,
                    fine_stage_start,
                )

        if show_progress:
            ShockFinder._print_progress("ShockFinder: linking finer AMR face neighbors", n, n, fine_stage_start)

        return neighbors, fine_neighbors

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
    def _pack_spatial(x: np.ndarray, y: np.ndarray, z: np.ndarray, spans: tuple[int, int, int]) -> np.ndarray:
        return (
            np.asarray(x, dtype=np.uint64) * np.uint64(spans[1]) * np.uint64(spans[2])
            + np.asarray(y, dtype=np.uint64) * np.uint64(spans[2])
            + np.asarray(z, dtype=np.uint64)
        )

    @staticmethod
    def _lookup_sorted(target_keys: np.ndarray, sorted_keys: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
        found = np.full(target_keys.shape, -1, dtype=np.int64)
        if sorted_keys.size == 0 or target_keys.size == 0:
            return found
        positions = np.searchsorted(sorted_keys, target_keys)
        valid = positions < sorted_keys.size
        if not np.any(valid):
            return found
        valid_positions = positions[valid]
        matched = sorted_keys[valid_positions] == target_keys[valid]
        valid_rows = np.nonzero(valid)[0]
        found[valid_rows[matched]] = sorted_indices[valid_positions[matched]]
        return found

    @staticmethod
    def _quantize(values: np.ndarray) -> np.ndarray:
        return np.floor(values + 0.5).astype(np.int64)

    @staticmethod
    def _quantize_inplace(values: np.ndarray) -> np.ndarray:
        np.add(values, 0.5, out=values)
        np.floor(values, out=values)
        return values.astype(np.int64)

    def _resolved_progress_interval(self, n: int) -> int:
        if self.progress_interval and self.progress_interval > 0:
            return int(self.progress_interval)
        return max(1, n // 10)

    def _progress(self, message: str) -> None:
        if self.show_progress:
            print(message, flush=True)

    @staticmethod
    def _print_progress(label: str, done: int, total: int, started_at: float | None = None) -> None:
        percent = 100.0 if total <= 0 else 100.0 * done / total
        if started_at is None:
            print(f"{label}: {done:,}/{total:,} ({percent:5.1f}%)", flush=True)
            return
        elapsed = max(0.0, time.perf_counter() - started_at)
        rate = done / elapsed if elapsed > 0.0 else 0.0
        print(
            f"{label}: {done:,}/{total:,} ({percent:5.1f}%) "
            f"elapsed={ShockFinder._format_elapsed(elapsed)} rate={rate:,.0f} cell/s",
            flush=True,
        )

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds_i = int(seconds)
        hours, rem = divmod(seconds_i, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes:d}m{secs:02d}s"
        return f"{seconds:.1f}s"
