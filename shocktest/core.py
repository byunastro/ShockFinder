from __future__ import annotations

import gc
import hashlib
import time
import warnings
from dataclasses import dataclass, fields
from typing import Any, Mapping
from pathlib import Path

import numpy as np

from . import _NATIVE_IMPORT_ERRORS, _shockfinder

_IMPORT_ERROR = _NATIVE_IMPORT_ERRORS.get("_shockfinder")

# These boolean diagnostics are losslessly reconstructed from the saved bitmask.
_VALIDATION_BITS = {
    'pressure_check_valid': 0, 'pressure_consistent': 1,
    'density_check_valid': 3, 'density_check_applicable': 4,
    'density_consistent': 5, 'mach_consistent': 7,
}


# Read/resave historical pickles without allocating these arrays in new results.
_LEGACY_RESULT_FIELDS = (
    'mach_temperature', 'mach_pressure', 'mach_density', 'temperature_ratio',
    'pressure_ratio', 'density_ratio', 'pressure_check_valid', 'pressure_consistent',
    'density_check_valid', 'density_consistent', 'density_check_applicable',
    'mach_consistent', 'mach_validation_status', 'temperature_floor',
)


class _HistoricalResultState:
    __slots__ = ('__dict__',)


@dataclass(slots=True)
class ShockResult(_HistoricalResultState):
    """One row per retained AMR cell.

    ``mach`` is the adopted pressure-jump Mach from profile reconstruction.
    ``pos`` retains leaf centers; ``center_position`` holds fitted shock centers.
    New results contain the adopted Mach and sparse physical quality diagnostics.
    """

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
    diagnostics: dict[str, int] | None = None
    gamma: float = 5.0 / 3.0
    position_unit: str = "km"
    temperature_unit: str = "K"
    density_unit: str = "Msol/kpc3"
    sampling_tolerance: float = 0.01
    # Unspecified/manual historical results retain their original interpretation;
    # every current find() result explicitly sets profile_pressure below.
    mach_estimator: str = "temperature_jump"
    pressure_unit: str = "erg/cm3"
    mean_molecular_weight: float = 0.59
    center_position: np.ndarray | None = None
    # Sparse, center-aligned reconstructed states. Cell indices are containing
    # leaves only; the measured states/positions can differ from leaf centers.
    sampling: dict[str, Any] | None = None

    def __getstate__(self):
        """Omit redundant flags from pickle, preserving independently edited flags.

        Check in bounded chunks to avoid full-length temporary arrays at save time.
        Explicit None values (disabled diagnostics) remain explicit.
        """
        state = {f.name: getattr(self, f.name) for f in fields(self)
                 if hasattr(self, f.name)}
        state.update(vars(self))
        status = state.get('mach_validation_status')
        if isinstance(status, np.ndarray) and status.ndim == 1:
            for name, bit in _VALIDATION_BITS.items():
                values = state.get(name)
                if (isinstance(values, np.ndarray) and values.dtype == np.dtype(bool)
                        and values.shape == status.shape
                        and all(np.array_equal(values[start:start+131072],
                            (status[start:start+131072] & (1 << bit)) != 0)
                            for start in range(0, status.size, 131072))):
                    del state[name]
        return None, state

    def __setstate__(self, state):
        # Accept the original slotted pickle state as well as the reduced state.
        values = state[1] if isinstance(state, tuple) else state
        for name, value in values.items():
            setattr(self, name, value)
        if 'center_position' not in values:self.center_position=None
        if 'mach_estimator' not in values:self.mach_estimator='temperature_jump'
        if 'pressure_unit' not in values:self.pressure_unit='density_unit K'
        if 'mean_molecular_weight' not in values:self.mean_molecular_weight=.59
        if 'sampling' not in values:
            self.sampling = None
        if 'temperature_unit' not in values:
            self.temperature_unit = 'K'
        if 'density_unit' not in values:
            self.density_unit = 'Msol/kpc3'
        if 'sampling_tolerance' not in values:
            self.sampling_tolerance = 0.01
        if not any(name in values for name in _LEGACY_RESULT_FIELDS):
            return
        status = values.get('mach_validation_status')
        for name, bit in _VALIDATION_BITS.items():
            if name not in values:
                restored = None
                if status is not None:
                    restored = np.empty(status.shape, dtype=bool)
                    for start in range(0, status.size, 131072):
                        restored[start:start+131072] = (status[start:start+131072] & (1 << bit)) != 0
                setattr(self, name, restored)

    def to_compact(self, **options):
        from .compact import compact_shocks
        return compact_shocks(self, **options)

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
        self.diagnostics = None
        self.__dict__.clear()
        self.sampling = None
        self.center_position = None
        gc.collect()


class ShockFinder:
    """AMR shock finder using local Profile reconstruction only."""

    def __init__(self) -> None:
        self.maxlevel = 20
        self.minlevel = 0
        self.gamma = 5.0 / 3.0
        self.min_temperature = None
        self.min_density = None
        self.max_density = None
        self.min_mach = 1.3
        self.sampling_tolerance = 0.01
        self.profile_batch_size = 32
        self.seed_bounds = None
        self.mean_molecular_weight = 0.59
        self.thermal_pressure_unit = 'erg/cm3'
        self.position_unit = "km"
        self.velocity_unit = "km/s"
        self.temperature_unit = "K"
        self.density_unit = "Msol/kpc3"
        self.show_progress = False
        self.progress_interval = 0
        self.boundary = "open"
        self.neighbor_backend = "fortran"
        self.neighbor_cache_dir: str | None = None
        # Exact integer narrowing is opt-in; floating-point physics is unchanged.
        self.index_dtype = "int64"
        # Set this to the exact mapping key (including a unit tuple, if used)
        # for a directly stored *thermal* pressure.  Automatic discovery is
        # deliberately limited to names that explicitly say "thermal" so that
        # magnetic, cosmic-ray, or turbulent pressure is not silently used.
        self.thermal_pressure_field: Any | None = None

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
        catalog_options: dict[str, Any] | None = None,
        dissipation_options: dict[str, Any] | None = None,
        compact: bool = False,
        compact_options: dict[str, Any] | None = None,
    ):
        """Run shock detection and optional post-processing in one pass.

        ``build_catalog=True`` calls ``shock_front_catalog`` with catalog_options
        and returns its 78-byte structured rows in analysis.catalog. Membership
        is always retained in analysis.labels, aligned with analysis.result.
        Defaults and grouping are identical to a separate call on those products.
        """

        if catalog_options is not None and not build_catalog:
            raise ValueError('catalog_options requires build_catalog=True')
        if compact_options is not None and not compact:
            raise ValueError("compact_options requires compact=True")
        result, neighbor_tables, timings = self._find_internal(cell)
        del neighbor_tables
        return self.analyze_result(result, cell=cell,
            compute_dissipation=compute_dissipation, build_catalog=build_catalog,
            catalog_options=catalog_options, dissipation_options=dissipation_options,
            compact=compact, compact_options=compact_options, timings=timings)

    def analyze_result(self, result, *, cell=None, compute_dissipation=True,
                       build_catalog=True, catalog_options=None,
                       dissipation_options=None, compact=False,
                       compact_options=None, timings=None):
        """Post-process a completed profile result without rerunning detection.

        New profile results contain all states needed by dissipation and catalogs;
        ``cell`` is needed only for historical result formats.
        """
        from .analysis import ShockAnalysis
        from .fronts import shock_front_catalog
        from .pyShockFinder import compute_dissipation as make_dissipation
        if catalog_options is not None and not build_catalog:
            raise ValueError('catalog_options requires build_catalog=True')
        if compact_options is not None and not compact:
            raise ValueError('compact_options requires compact=True')
        timings = {} if timings is None else dict(timings)
        analysis_start = time.perf_counter() - timings.get('detection_total', 0.)
        # Catalogs use the same saved-geometry path, with or without analyze.
        dissipation = None
        catalog = None
        labels = None
        if compute_dissipation:
            stage_start = time.perf_counter()
            options = {} if dissipation_options is None else dict(dissipation_options)
            if "gamma" in options and not np.isclose(float(options["gamma"]), float(self.gamma)):
                raise ValueError(
                    "dissipation gamma must match finder.gamma for a physically "
                    "consistent single-pass analysis"
                )
            if "temperature_floor" in options:
                raise ValueError("temperature_floor is removed; profile states are not floored")
            options["gamma"] = float(self.gamma)
            dissipation = make_dissipation(
                cell, result, _compact=compact, **options
            )
            timings["dissipation"] = time.perf_counter() - stage_start
        options = {} if catalog_options is None else dict(catalog_options)
        if 'return_labels' in options:
            raise ValueError('analyze always stores membership; omit return_labels')
        if compact:
            # Group the independent compact product so saved validation flags
            # are available without expanding diagnostics or labels to the mesh.
            from .compact import compact_shocks
            compact_start = time.perf_counter()
            samples = compact_shocks(
                result, dissipation, timings=timings,
                _compact_dissipation=True, **(compact_options or {}),
            )
            timings['compact'] = time.perf_counter() - compact_start
            result.clear()
            if dissipation is not None:
                dissipation.clear()
            if build_catalog:
                stage_start = time.perf_counter()
                samples.groups, samples.columns['group_id'] = shock_front_catalog(
                    samples, return_labels=True, **options)
                timings['catalog'] = time.perf_counter() - stage_start
                samples.metadata['counts']['groups'] = len(samples.groups)
                samples.metadata['counts']['representative'] = int(samples.groups['ncell'].sum(dtype=np.int64))
                samples.metadata['catalog'] = {'schema': 'front_dtype', 'bytes_per_row': 78}
            timings['total'] = time.perf_counter() - analysis_start
            samples.timings = timings
            return samples
        if build_catalog:
            stage_start = time.perf_counter()
            catalog, labels = shock_front_catalog(
                result, dissipation, return_labels=True, **options
            )
            timings["catalog"] = time.perf_counter() - stage_start
        timings["total"] = time.perf_counter() - analysis_start
        return ShockAnalysis(result=result, dissipation=dissipation, catalog=catalog,
                             timings=timings, labels=labels)

    def clear(self) -> None:
        """Compatibility cleanup hook.

        ``ShockFinder`` does not keep large AMR arrays after each run; those are
        owned by the returned ``ShockResult``. Call ``result.clear()`` to release
        result arrays explicitly.
        """

        gc.collect()

    def find(self, cell: Any, *, compact: bool = False, compact_options=None):
        if compact:
            return self.analyze(cell, compute_dissipation=False, build_catalog=False,
                                compact=True, compact_options=compact_options)
        if compact_options is not None:
            raise ValueError("compact_options requires compact=True")
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
        self._progress("ShockFinder: reading AMR cell fields")
        stage_start = time.perf_counter()
        arrays = self._extract_amr_arrays(cell)
        timings["input"] = time.perf_counter() - stage_start
        selected_indices = arrays.pop("selected_indices")
        n = arrays["temp"].size
        self._progress(f"ShockFinder: retained {n:,} cells after filtering")

        from .profile_reconstruction import detect
        fits=[];diagnostics={}
        if n:
            pressure,inferred=self._thermal_pressure_values(cell,selected_indices,arrays)
            fits,diagnostics=detect(arrays,self,pressure)
        else:
            pressure=None
        result = self._assemble_result(arrays, selected_indices, fits, diagnostics)
        timings['scan']=time.perf_counter()-total_start-timings['input']
        timings['neighbors']=0.;timings['detection_total']=time.perf_counter()-total_start
        return result,(None,None,None),timings

    def _assemble_result(self, arrays, selected_indices, fits, diagnostics):
        """Assemble once after serial or distributed profile reduction."""
        n = len(arrays['dx'])
        rows=np.array([f['cell'] for f in fits],dtype=np.int64)
        mach=np.zeros(n);shock=np.zeros(n,dtype=bool);normal=np.zeros((n,3));zone=np.zeros(n)
        index_type=np.int32 if self.index_dtype=='auto' and n<=np.iinfo(np.int32).max else np.int64
        centers=np.full(n,-1,dtype=index_type);up=centers.copy();down=centers.copy()
        positions=np.full((n,3),np.nan)
        if len(rows):
            mach[rows]=[f['mach'] for f in fits];shock[rows]=True;centers[rows]=rows
            up[rows]=[f['endpoints'][0] for f in fits];down[rows]=[f['endpoints'][1] for f in fits]
            normal[rows]=[f['normal'] for f in fits];positions[rows]=[f['point'] for f in fits]
            zone[rows]=[np.linalg.norm(f['support_pos'][1]-f['support_pos'][0]) for f in fits]
        sampling={
            'rows':rows,'temperature':np.array([f['states_temperature'] for f in fits]).reshape(-1,2),
            'density':np.array([f['states_density'] for f in fits]).reshape(-1,2),
            'pressure':np.array([f['states_pressure'] for f in fits]).reshape(-1,2),
            'pos':np.repeat(positions[rows,None,:],2,axis=1),
            'distance':np.array([[np.linalg.norm(f['point']-q) for q in f['support_pos']] for f in fits]).reshape(-1,2),
            'support_pos':np.array([f['support_pos'] for f in fits]).reshape(-1,2,3),
            'converged':np.ones(len(rows),dtype=bool),
            'reference_width':np.array([f['reference_width'] for f in fits]),
            'density_residual':np.array([f['density_residual'] for f in fits]),
            'velocity_residual':np.array([f['velocity_residual'] for f in fits]),
            'center_constraint_active':np.array([f['center_constraint_active'] for f in fits],dtype=bool),
            'mass_surface_displacement':np.array([f['mass_surface_displacement']*f['reference_width'] for f in fits]),
            'profile_passes':np.array([f['passes'] for f in fits],dtype=np.int32),
            'quality':np.array([f['quality'] for f in fits]),
            'background_orders':np.array([f['orders'] for f in fits],dtype=np.int32).reshape(-1,4),
        }
        # Sparse diagnostics only: never allocate per-cell uncertainty arrays.
        if not fits or 'quality_flags' in fits[0]:
            for key in ('entropy_margin','center_sensitivity','momentum_residual','energy_residual',
                        'shock_speed_mass','shock_speed_pressure'):
                sampling[key]=np.array([f[key]*(f['reference_width'] if key=='center_sensitivity' else 1) for f in fits])
            sampling['velocity']=np.array([f['states_velocity'] for f in fits]).reshape(-1,2,3)
            sampling['mach_model_range']=np.array([f['mach_range'] for f in fits]).reshape(-1,2)
            sampling['center_model_range']=np.array([(f['offset_range']-f['offset'])*f['reference_width'] for f in fits]).reshape(-1,2)
            sampling['background_model_count']=np.array([f['model_count'] for f in fits],dtype=np.int32)
            sampling['profile_quality_flags']=np.array([f['quality_flags'] for f in fits],dtype=np.uint16)
            sampling['quality_clear']=sampling['profile_quality_flags']==0
        result=ShockResult(mach=mach,shock=shock,center_index=centers,upstream_index=up,
                           downstream_index=down,selected_indices=selected_indices,
                           pos=arrays['pos'],center_position=positions,dx=arrays['dx'],normal=normal,
                           level=arrays['level'],zone_width=zone,sampling=sampling,diagnostics=diagnostics)
        result.gamma=float(self.gamma)
        result.position_unit=self.position_unit;result.temperature_unit=self.temperature_unit
        result.density_unit=self.density_unit;result.sampling_tolerance=self.sampling_tolerance
        result.mach_estimator='profile_pressure'
        result.mean_molecular_weight=float(self.mean_molecular_weight)
        return result

    def _thermal_pressure_values(
        self,
        cell: Any,
        selected_indices: np.ndarray,
        arrays: dict[str, np.ndarray],
        *,
        dtype=np.float64,
        endpoint_rows=None,
    ) -> tuple[np.ndarray | None, bool]:
        """Return selected thermal pressure, or signal the rho*T fallback."""

        key = self.thermal_pressure_field
        value = None
        if key is not None:
            try:
                value = self._get(cell, key)
            except (KeyError, TypeError, IndexError) as exc:
                raise KeyError(
                    f"cell is missing configured thermal pressure field {key!r}"
                ) from exc
        elif isinstance(cell, Mapping):
            explicit_names = {
                "thermal_pressure",
                "pressure_thermal",
                "p_thermal",
                "pth",
            }
            for candidate, candidate_value in cell.items():
                name = candidate[0] if isinstance(candidate, tuple) else candidate
                if isinstance(name, str) and name.lower() in explicit_names:
                    value = candidate_value
                    break

        if value is None:
            return None, True
        pressure = np.asarray(value)
        if pressure.ndim != 1:
            raise ValueError("thermal pressure must be a 1D AMR cell field")
        if selected_indices.size and np.max(selected_indices) >= pressure.size:
            raise ValueError("thermal pressure field is shorter than the cell table")
        if not selected_indices.size:
            return np.empty(0, dtype=np.float64), False
        source_rows = selected_indices if endpoint_rows is None else selected_indices[endpoint_rows]
        return np.asarray(pressure[source_rows], dtype=dtype), False

    def _neighbor_tables_for_arrays(
        self,
        pos,
        dx,
        level,
        *,
        show_progress=False,
        progress_interval=0,
    ):
        if self.neighbor_cache_dir is None:
            return self._build_neighbor_tables(
                pos,
                dx,
                level,
                show_progress=show_progress,
                progress_interval=progress_interval,
                backend=self.neighbor_backend,
            )

        cache_dir = Path(self.neighbor_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = self._geometry_digest(pos, dx, level)
        prefix = cache_dir / f"neighbors-{digest}"
        paths = (
            Path(f"{prefix}-same.npy"),
            Path(f"{prefix}-fine-index.npy"),
            Path(f"{prefix}-fine-values.npy"),
        )
        if all(path.exists() for path in paths):
            return tuple(
                np.load(path, mmap_mode="r", allow_pickle=False) for path in paths
            )

        tables = self._build_neighbor_tables(
            pos,
            dx,
            level,
            show_progress=show_progress,
            progress_interval=progress_interval,
            backend=self.neighbor_backend,
        )
        for path, values in zip(paths, tables):
            np.save(path, np.asfortranarray(values), allow_pickle=False)
        return tables

    @staticmethod
    def _geometry_digest(pos, dx, level):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(b"shockfinder-neighbors-sparse-v1")
        for values in (pos[:, 0], pos[:, 1], pos[:, 2], dx, level):
            array = np.ascontiguousarray(values)
            digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    def _validate_settings(self) -> None:
        if (isinstance(self.sampling_tolerance, (bool, np.bool_))
                or not np.isfinite(self.sampling_tolerance)
                or not 0.0 <= self.sampling_tolerance < 1.0):
            raise ValueError("sampling_tolerance must be finite and in [0, 1)")
        if self.index_dtype not in ("int64", "auto"):
            raise ValueError("index_dtype must be 'int64' or 'auto'")
        if not np.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("gamma must be finite and greater than 1")
        if not np.isfinite(self.min_mach) or self.min_mach < 1.0:
            raise ValueError("min_mach must be finite and at least 1")
        if not np.isfinite(self.mean_molecular_weight) or self.mean_molecular_weight<=0:
            raise ValueError('mean_molecular_weight must be positive')
        if isinstance(self.profile_batch_size,bool) or int(self.profile_batch_size)!=self.profile_batch_size or self.profile_batch_size<1:
            raise ValueError('profile_batch_size must be a positive integer')
        for name in ('center_normal_cosine','center_plateau_tolerance','max_center_steps','max_steps','method'):
            if hasattr(self,name):raise ValueError(f'{name} is removed: Profile reconstruction is the only estimator')
        for name in ('temperature_floor','validate_mach','filter_inconsistent','consistency_factor',
                     'density_check_max_mach','density_saturation_rtol','mach_validation_dtype'):
            if hasattr(self,name):
                raise ValueError(f'{name} is removed: use unfloored profile states and profile quality diagnostics')
        if int(self.minlevel) > int(self.maxlevel):
            raise ValueError("minlevel must not exceed maxlevel")
        if self.neighbor_backend not in {"fortran", "numpy"}:
            raise ValueError("neighbor_backend must be 'fortran' or 'numpy'")
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

        # Keep the complete level-selected geometry. Temperature and density
        # cuts only control which cells may start a shock detection; filtered
        # cells remain available to the reconstructed profiles.
        mask = (level >= int(self.minlevel)) & (level <= int(self.maxlevel))
        selected_indices = np.nonzero(mask)[0].astype(np.int64, copy=False)
        if self.index_dtype == "auto" and n <= np.iinfo(np.int32).max:
            selected_indices = selected_indices.astype(np.int32)
        del mask

        selected_temp = temp[selected_indices]
        selected_rho = rho[selected_indices]
        detection_mask = np.ones(selected_indices.size, dtype=np.int32)
        if self.min_temperature is not None:
            detection_mask[selected_temp < float(self.min_temperature)] = 0
        if self.min_density is not None:
            detection_mask[selected_rho < float(self.min_density)] = 0
        if self.max_density is not None:
            detection_mask[selected_rho > float(self.max_density)] = 0

        if self.seed_bounds is not None:
            bounds=np.asarray(self.seed_bounds,dtype=float)
            if bounds.shape!=(3,2) or not np.isfinite(bounds).all() or np.any(bounds[:,1]<=bounds[:,0]):
                raise ValueError('seed_bounds must have shape (3,2) with increasing physical-coordinate bounds')
            coords=np.column_stack((x[selected_indices],y[selected_indices],z[selected_indices]))
            detection_mask[~np.all((coords>=bounds[:,0])&(coords<=bounds[:,1]),axis=1)]=0

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
        selected_dx = np.asfortranarray(dx[selected_indices], dtype=np.float64)
        selected_level = np.asfortranarray(level[selected_indices], dtype=np.int32)
        selected_temp = np.asfortranarray(selected_temp, dtype=np.float64)
        selected_rho = np.asfortranarray(selected_rho, dtype=np.float64)
        for name, values in (
            ("position", pos),
            ("velocity", vel),
            ("dx", selected_dx),
            ("T", selected_temp),
            ("rho", selected_rho),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        return {
            "pos": pos,
            "vel": vel,
            "dx": selected_dx,
            "temp": selected_temp,
            "rho": selected_rho,
            "level": selected_level,
            "detection_mask": np.asfortranarray(detection_mask, dtype=np.int32),
            "selected_indices": selected_indices,
        }

    @staticmethod
    def _to_python_indices(values: np.ndarray, *, mode="int64") -> np.ndarray:
        # f2py emits signed int32 mesh indices. Keep that lossless representation
        # when requested; do not narrow wider external values without checking.
        dtype = np.int32 if mode == "auto" and values.dtype == np.int32 else np.int64
        out = np.array(values, dtype=dtype, copy=True)
        out -= 1
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
        neighbors, _, _ = ShockFinder._build_neighbor_tables(
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
        backend: str = "fortran",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if backend == "numpy":
            return ShockFinder._build_neighbor_tables_numpy(
                pos,
                dx,
                level,
                show_progress=show_progress,
                progress_interval=progress_interval,
            )
        if backend != "fortran":
            raise ValueError("neighbor backend must be 'fortran' or 'numpy'")
        if _shockfinder is None:
            raise ImportError("shocktest Fortran extension is not built") from _IMPORT_ERROR
        n = int(np.asarray(dx).size)
        if n == 0:
            return (
                np.zeros((0, 6), dtype=np.int32, order="F"),
                np.zeros((0, 6), dtype=np.int32, order="F"),
                np.zeros((0, 4), dtype=np.int32, order="F"),
            )
        dx_values = np.asarray(dx, dtype=np.float64)
        if np.any(~np.isfinite(dx_values)) or np.any(dx_values <= 0.0):
            raise ValueError("dx must be finite and positive for every retained cell")
        unique_widths = np.unique(dx_values)
        if unique_widths.size > 1 and np.any(
            unique_widths[1:] / unique_widths[:-1] > 2.0 * (1.0 + 1.0e-12)
        ):
            raise ValueError(
                "fine AMR face neighbors currently support one refinement jump "
                "(2x per axis, 4 cells per face). Include intermediate AMR levels."
            )
        pos_f = np.asfortranarray(pos, dtype=np.float64)
        dx_f = np.asfortranarray(dx_values, dtype=np.float64)
        level_f = np.asfortranarray(level, dtype=np.int32)
        neighbors, fine_face_index, nfine = (
            _shockfinder.shockfinder_kernel.build_neighbor_index(
                pos_f, dx_f, level_f, n
            )
        )
        fine_neighbors = _shockfinder.shockfinder_kernel.fill_fine_neighbors(
            pos_f, dx_f, level_f, fine_face_index, int(nfine), n
        )
        return (
            np.asfortranarray(neighbors, dtype=np.int32),
            np.asfortranarray(fine_face_index, dtype=np.int32),
            np.asfortranarray(fine_neighbors, dtype=np.int32),
        )

    @staticmethod
    def _build_neighbor_tables_numpy(
        pos: np.ndarray,
        dx: np.ndarray,
        level: np.ndarray,
        *,
        show_progress: bool = False,
        progress_interval: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = dx.size
        neighbors = np.zeros((n, 6), dtype=np.int32, order="F")
        fine_neighbors = np.zeros((n, 6, 4), dtype=np.int32, order="F")
        if n == 0:
            return (
                neighbors,
                np.zeros((0, 6), dtype=np.int32, order="F"),
                np.zeros((0, 4), dtype=np.int32, order="F"),
            )

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

        fine_face_mask = np.any(fine_neighbors > 0, axis=2)
        fine_rows, fine_faces = np.nonzero(fine_face_mask)
        fine_face_index = np.zeros((n, 6), dtype=np.int32, order="F")
        if fine_rows.size:
            fine_face_index[fine_rows, fine_faces] = np.arange(
                1, fine_rows.size + 1, dtype=np.int32
            )
            sparse_fine_neighbors = np.asfortranarray(
                fine_neighbors[fine_rows, fine_faces, :], dtype=np.int32
            )
        else:
            sparse_fine_neighbors = np.zeros((0, 4), dtype=np.int32, order="F")
        return neighbors, fine_face_index, sparse_fine_neighbors

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
