"""Build, inspect, save, and reload a ShockFinder surface catalog.

This example keeps the project's input convention: ``cell`` is an AMR cell
table extracted from part of one simulation snapshot. The extraction itself is
simulation-reader specific and should be performed before calling
``make_shock_catalog``.

``merger_shock_catalog`` instead accepts saved ShockFinder result/dissipation
objects and a cluster-center history. It performs post-processing only and
returns existing dense shock-center indices, evidence, and reversible front
membership. No detector or file I/O is invoked by that function.

Required fields are documented in the project README. In particular, positions
and cell widths are in km, velocities in km/s, temperature in K, and density in
Msol/kpc3. Extracted regions always use an open boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt

import shocktest

try:
    from shocktest import _merger_neighbors
except ImportError:
    _merger_neighbors = None

_MERGER_KPC_KM = 3.0856775814913673e16
_MERGER_KPC_PER_GYR_TO_KMS = _MERGER_KPC_KM / (1.0e9 * 365.25 * 86400.0)
_MERGER_LENGTH_TO_KPC = {"km": 1.0 / _MERGER_KPC_KM, "kpc": 1.0, "Mpc": 1000.0}
_MERGER_SCHEMA_VERSION = 2
_MERGER_EVIDENCE_NAMES = ("epoch", "extent", "normal", "axis", "continuity", "outward_motion", "dissipation", "origin")


def merger_shock_catalog(iout, result, dissipation, cluster_info):
    """Identify plausible merger shocks using already computed dense results.

    Parameters
    ----------
    iout : int
        Snapshot being classified.
    result, dissipation
        Saved ``ShockResult`` and ``DissipationResult`` objects or mappings
        with the same fields. They must have matching retained-cell rows.
    cluster_info : dict
        Arrays covering the configured snapshot range:
        ``iout`` (N,), ``ccen1``/``ccen2`` (N,3), ``t_BB`` (N,) in Gyr,
        ``redshift`` (N,), optional ``rvir1``/``rvir2`` (N,).
        ``snapshot`` aliases ``iout``; ``time_gyr`` aliases ``t_BB``;
        ``cluster_rvir``/``cluster_rvir2`` alias the two radii.
        Centers/radii default to physical kpc, as in the supplied NewCluster
        RUR conversion. Declare ``position_unit`` for km or Mpc instead.
        Fill secondary coordinates/radius with NaN after its branch ends.
        Virial masses may be included but are not required by this method.

        ``previous_catalog`` optionally contains the previous return value;
        pass it when advancing through outputs to measure propagation.
        ``merger_shock_options`` optionally overrides documented thresholds.
        Performance options: ``cell_chunk_size`` (131072),
        ``spatial_query_chunk`` (5000), ``max_neighbor_pairs`` (200000).
        ``neighbor_backend`` is 'auto' (compiled Fortran when available),
        'scipy', or 'fortran'. The separate optional post-processing extension
        is built from shocktest/fortran/merger_neighbors.f90; it never calls
        ShockFinder. Fortran uses spatial bins with the same exact AMR cuts;
        unsupported bin ranges fall back to SciPy search + Fortran pair merging.
        Set ``expand_candidate_cells=False`` to return representative EXISTING
        center IDs and assessments once per candidate front; complete reversible
        cell membership remains available. Default True preserves the old API.
        This dictionary and both ShockFinder inputs are not modified.

    Returns
    -------
    dict
        ``shock_id``, ``evidence``, ``confidence``, ``quality_flags``, and
        ``uncertain`` are aligned arrays/tuples for candidate center cells.
        ``shock_id`` uses the EXISTING ``result.center_index`` (0-based dense
        retained-row space): ``result.pos[shock_id]`` and
        ``result.mach[shock_id]`` directly recover those measurements.

        ``fronts`` holds compact assessments for all valid fronts, including
        uncertain/unclassified ones. ``membership`` maps every detected
        center to its front (``front_index=-1`` for invalid detections), and
        preserves original input rows via ``input_cell_id``. Fronts use an
        existing member center index as their representative, not a new ID.
        ``tracking_state`` stores only front summaries for subsequent calls.
        No dense result/dissipation object is retained.

        Evidence is an index in [0,1], not a probability or proof of origin.
        An isolated call has unmeasured propagation. After the sequence, call
        ``finalize_merger_shock_catalogs`` to update earlier provisional scores.
        ``timings_seconds`` reports analysis stages, excluding input file I/O.
        A ``MergerShockInputs`` cache can replace result with dissipation=None.
    """
    started = time.perf_counter()
    timings = {}
    if isinstance(iout, bool) or not isinstance(iout, (int, np.integer)):
        raise ValueError("iout must be an integer snapshot number")
    iout = int(iout)
    config = _merger_config(cluster_info)
    centers, meta, history, signature = _merger_center_history(iout, cluster_info, config)
    previous = cluster_info.get("previous_catalog")
    if previous is None:
        frames = ()
        epochs = _merger_verify_epochs(config, centers, meta, history)
        geoms = _merger_geometry_history(centers, meta, history, config)
    else:
        state = previous.get("tracking_state", {})
        if state.get("schema_version") != _MERGER_SCHEMA_VERSION:
            raise ValueError("previous_catalog is not a compatible merger catalog")
        if not _merger_same_scientific_options(state["configuration"], vars(config)) or state["center_history_signature"] != signature:
            raise ValueError("center history or options changed; start a new sequence with previous_catalog=None")
        frames = state["frames"]
        if frames and iout <= frames[-1]["snapshot"]:
            raise ValueError("calls with previous_catalog must advance in snapshot order")
        # Reuse immutable history metadata. No retained-cell arrays are shared
        # with the inputs, and no earlier state is updated during scoring.
        meta, epochs = state["metadata"], state["epochs"]
        geoms = state.get("geometry_history") or _merger_geometry_history(centers, meta, history, config)
    timings["metadata"] = time.perf_counter()-started
    stage = time.perf_counter()
    data = _merger_compact_results(result, dissipation, snapshot=iout, chunk_size=config.cell_chunk_size)
    n = data["retained_count"]
    timings["compact"] = time.perf_counter()-stage
    stage = time.perf_counter()
    groups = _merger_group_cells(data, config)
    timings["group"] = time.perf_counter()-stage
    current = []
    labels = np.full(len(data["retained_row"]), -1, dtype=np.int32)
    fingerprints = []
    timings["summarize"], timings["fingerprint"] = 0., 0.
    for group_index, rows in enumerate(groups):
        stage = time.perf_counter()
        front = _merger_summarize_front(rows, data, geoms[iout], iout, meta[iout]["time_gyr"], meta[iout]["redshift"], config)
        front["representative_shock_id"] = int(np.min(data["retained_row"][rows]))
        front["base_quality_flags"] = tuple(front["quality_flags"])
        labels[rows] = group_index
        timings["summarize"] += time.perf_counter()-stage
        stage = time.perf_counter()
        fingerprints.append(_merger_group_fingerprint(rows, data, config.cell_chunk_size))
        timings["fingerprint"] += time.perf_counter()-stage
        del front["rows"]  # No member arrays or dense inputs in tracking state.
        current.append(front)
    del groups
    frames = frames + ({"snapshot": iout, "fronts": current,
                        "source_fingerprint": tuple(fingerprints)},)
    state = {"schema_version": _MERGER_SCHEMA_VERSION, "configuration": vars(config),
             "center_history_signature": signature, "metadata": meta, "epochs": epochs,
             "geometry_history": geoms, "frames": frames}
    stage = time.perf_counter()
    scored, diss_scale = _merger_score_sequence(state)
    timings["score_and_track"] = time.perf_counter()-stage
    catalog = {"schema_version": _MERGER_SCHEMA_VERSION, "iout": iout,
        "fronts": _merger_public_fronts(scored[iout]),
        "membership": {"shock_id": data["retained_row"], "input_cell_id": data["cell_id"],
                       "front_index": labels, "source_fingerprint": tuple(fingerprints)},
        "source": {"retained_count": n, "position_unit": data["result_position_unit"],
                   "id_space": "dense retained result row / center_index", "snapshot": iout},
        "epoch_verification": epochs, "dissipation_reference_p75_erg_s": diss_scale,
        "neighbor_backend": _merger_neighbor_backend(data, config),
        "tracking_state": state, "timings_seconds": timings,
    }
    stage = time.perf_counter()
    _merger_selection(catalog)
    timings["selection"] = time.perf_counter()-stage
    timings["total"] = time.perf_counter()-started
    return catalog


def get_merger_shock_members(catalog, shock_id, result, dissipation=None):
    """Recover the entire front containing an existing center ID.

    For dense results, geometry is in their original position_unit. A
    MergerShockInputs cache returns physical kpc and unit normals, with saved
    endpoint positions when included. Endpoint indices and shock IDs always
    remain in ORIGINAL dense retained-row space; -1 means unavailable.
    """
    if isinstance(shock_id, bool) or not isinstance(shock_id, (int, np.integer)):
        raise ValueError("shock_id must be an integer dense center index")
    membership = catalog["membership"]
    rows = membership["shock_id"]
    match = np.flatnonzero(rows == shock_id)
    if len(match) != 1:
        raise KeyError(f"shock center {shock_id} is not in snapshot {catalog['iout']}")
    group = int(membership["front_index"][match[0]])
    if group < 0:
        raise ValueError("this detection failed saved-field validity selection and has no front")
    member_mask = membership["front_index"] == group
    member_rows = rows[member_mask]
    if isinstance(result, MergerShockInputs):
        if dissipation is not None:
            raise ValueError("cached inputs already include saved dissipation; pass None")
        data = result.data
        if (result.snapshot != catalog["iout"] or data["retained_count"] != catalog["source"]["retained_count"]
                or data["result_position_unit"] != catalog["source"]["position_unit"]):
            raise ValueError("cache source does not match this catalog")
        take = np.searchsorted(data["retained_row"], member_rows)
        if (np.any(take >= len(data["retained_row"])) or not np.array_equal(data["retained_row"][take], member_rows)
                or not np.array_equal(data["cell_id"][take], membership["input_cell_id"][member_mask])):
            raise ValueError("cached member indices do not match this catalog")
        signature = _merger_group_fingerprint(take, data, 131072)
        if signature != membership["source_fingerprint"][group]:
            raise ValueError("cache member measurements do not match this catalog")
        return {"iout": result.snapshot, "shock_id": member_rows.copy(), "input_cell_id": data["cell_id"][take],
                "pos": data["pos"][take], "dx": data["dx"][take], "normal": data["normal"][take],
                "mach": data["mach"][take], "flux": data["flux"][take], "total": data["total"][take],
                "area": data["area"][take], "position_unit": "kpc", "original_position_unit": data["result_position_unit"],
                "assessment": dict(catalog["fronts"][group]),
                "dissipation_units": {"flux": "erg/s/kpc2", "total": "erg/s", "area": "kpc2"},
                **{name: array[take] for name, array in result.extras.items()}}
    n = len(_merger_required(result, "mach"))
    unit = _merger_required(result, "position_unit")
    if n != catalog["source"]["retained_count"] or unit != catalog["source"]["position_unit"]:
        raise ValueError("result shape or position unit does not match the catalog source")
    ids = _merger_array(result, "selected_indices", n)[member_rows]
    if (not np.array_equal(ids, membership["input_cell_id"][member_mask]) or
            not np.all(_merger_array(result, "shock", n)[member_rows]) or
            not np.array_equal(_merger_array(result, "center_index", n)[member_rows], member_rows)):
        raise ValueError("result indices do not match the catalog source")
    pos = _merger_array(result, "pos", shape=(n, 3))[member_rows]
    mach = _merger_array(result, "mach", n)[member_rows]
    dx = _merger_array(result, "dx", n)[member_rows]
    normal = _merger_array(result, "normal", shape=(n, 3))[member_rows]
    float_normal = normal.astype(float)
    norm = np.linalg.norm(float_normal, axis=1)
    normalized = float_normal / np.maximum(norm[:, None], 1e-300)
    signature = _merger_source_fingerprint(member_rows, ids, pos.astype(float) * _MERGER_LENGTH_TO_KPC[unit],
                                    mach, dx.astype(float) * _MERGER_LENGTH_TO_KPC[unit], normalized)
    if signature != membership["source_fingerprint"][group]:
        raise ValueError("result member measurements do not match the catalog source")
    recovered = {"iout": catalog["iout"], "shock_id": member_rows.copy(),
                 "input_cell_id": np.asarray(ids).copy(), "pos": pos, "mach": mach,
                 "dx": dx, "normal": normal, "position_unit": unit,
                 "assessment": dict(catalog["fronts"][group])}
    for name in ("center_index", "upstream_index", "downstream_index", "level", "zone_width",
                 "mach_consistent", "mach_validation_status"):
        if _merger_value(result, name) is not None:
            recovered[name] = _merger_array(result, name, n)[member_rows]
    if dissipation is not None:
        for name in ("flux", "total", "area"):
            recovered[name] = _merger_array(dissipation, name, n)[member_rows]
        recovered["dissipation_units"] = {"flux": "erg/s/kpc2", "total": "erg/s", "area": "kpc2"}
    return recovered


def finalize_merger_shock_catalogs(catalogs):
    """Refresh earlier assessments using the final sequence's temporal evidence.

    No cell measurements are reloaded. The returned dictionaries have new
    assessment tables and reuse their unchanged membership arrays. Inputs are
    not mutated. Empty input returns an empty list.
    """
    catalogs = list(catalogs)
    if not catalogs:
        return []
    state = catalogs[-1]["tracking_state"]
    started = time.perf_counter()
    scored, diss_scale = _merger_score_sequence(state)
    frames = {frame["snapshot"]: frame for frame in state["frames"]}
    refreshed = []
    for catalog in catalogs:
        own_state = catalog["tracking_state"]
        if (not _merger_same_scientific_options(own_state["configuration"], state["configuration"]) or
                own_state["center_history_signature"] != state["center_history_signature"]):
            raise ValueError("catalogs must belong to one unchanged analysis sequence")
        fronts = scored.get(catalog["iout"])
        if fronts is None or [f["representative_shock_id"] for f in fronts] != [f["shock_id"] for f in catalog["fronts"]]:
            raise ValueError("catalog is not represented in the final tracking state")
        if catalog["membership"]["source_fingerprint"] != frames[catalog["iout"]]["source_fingerprint"]:
            raise ValueError("catalog cell measurements differ from the final sequence")
        updated = dict(catalog)
        updated["fronts"] = _merger_public_fronts(fronts)
        updated["tracking_state"] = state
        updated["dissipation_reference_p75_erg_s"] = diss_scale
        updated["timings_seconds"] = dict(catalog.get("timings_seconds", {}))
        refreshed.append(_merger_selection(updated))
    elapsed = time.perf_counter()-started
    for updated in refreshed:
        updated["timings_seconds"]["finalize_sequence"] = elapsed
    return refreshed


@dataclass
class _MergerOptions:
    """Configurable merger evidence scales; all lengths are physical kpc.

    Default score cutoffs are starting values, not calibrated probabilities.
    Cells and fronts below individual criteria remain recorded with flags.
    Override these fields through cluster_info["merger_shock_options"].
    """
    center_position_unit: str = "kpc"
    center_coordinate_frame: str = "physical"
    result_coordinate_frame: str = "physical"
    dissipation_flux_unit: str = "erg/s/kpc2"
    dissipation_total_unit: str = "erg/s"
    area_unit: str = "kpc2"
    max_cell_gap_factor: float = 0.25
    minimum_neighbor_normal_cosine: float = 0.6
    minimum_front_cells: int = 3
    cluster_extent_kpc: float = 300.0
    axis_offset_scale_kpc: float = 400.0
    origin_scale_kpc: float = 500.0
    passage_window_gyr: float = 0.5
    max_link_interval_gyr: float = 0.3
    max_front_speed_kpc_gyr: float = 3000.0
    minimum_track_length: int = 3
    candidate_score: float = 0.65
    uncertain_score: float = 0.35
    thresholds_calibrated: bool = False
    reference_snapshot_tolerance: int = 20
    reference_redshift_tolerance: float = 0.05
    spatial_query_chunk: int = 5000
    cell_chunk_size: int = 131072
    max_neighbor_pairs: int = 200000
    neighbor_backend: str = "auto"
    expand_candidate_cells: bool = True
    box_size_kpc: float | None = None
    post_merger_axis_policy: str = "last_measured"
    reference_epochs: dict = field(default_factory=lambda: {
        "overlap": {"snapshot": 605, "redshift": 0.85},
        "pericenter": {"snapshot": 710, "redshift": 0.67},
        "apocenter": {"snapshot": 785, "redshift": 0.58},
    })

    def validate(self):
        """Validate scientific options for the object interface."""
        config = self
        for key in ("center_coordinate_frame", "result_coordinate_frame"):
            if getattr(config, key) != "physical":
                raise ValueError(f"{key} must be explicitly 'physical'; convert comoving/code coordinates before analysis")
        if config.center_position_unit not in _MERGER_LENGTH_TO_KPC:
            raise ValueError("center_position_unit must be km, kpc, or Mpc")
        expected_units = {"dissipation_flux_unit": "erg/s/kpc2", "dissipation_total_unit": "erg/s", "area_unit": "kpc2"}
        for name, unit in expected_units.items():
            if getattr(config, name) != unit:
                raise ValueError(f"{name} must be declared as {unit!r}; convert saved fields explicitly for other units")
        for name in ("cluster_extent_kpc", "axis_offset_scale_kpc", "origin_scale_kpc", "passage_window_gyr", "max_link_interval_gyr", "max_front_speed_kpc_gyr"):
            if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(config.max_cell_gap_factor) or config.max_cell_gap_factor < 0:
            raise ValueError("max_cell_gap_factor must be nonnegative and finite")
        if config.minimum_front_cells < 1 or config.minimum_track_length < 1 or config.spatial_query_chunk < 1:
            raise ValueError("cell, track, and query counts must be positive")
        for name in ("cell_chunk_size", "max_neighbor_pairs", "spatial_query_chunk"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(config.expand_candidate_cells, (bool, np.bool_)):
            raise ValueError("expand_candidate_cells must be boolean")
        if config.neighbor_backend not in {"auto", "scipy", "fortran"}:
            raise ValueError("neighbor_backend must be 'auto', 'scipy', or 'fortran'")
        if not 0 <= config.minimum_neighbor_normal_cosine <= 1:
            raise ValueError("minimum_neighbor_normal_cosine must be in [0, 1]")
        if not 0 <= config.uncertain_score < config.candidate_score <= 1:
            raise ValueError("score cutoffs must satisfy 0 <= uncertain < candidate <= 1")
        if config.box_size_kpc is not None and config.box_size_kpc <= 0:
            raise ValueError("box_size_kpc must be positive")
        if config.post_merger_axis_policy not in {"last_measured", "unavailable"}:
            raise ValueError("post_merger_axis_policy must be 'last_measured' or 'unavailable'")
        return self


@dataclass(frozen=True)
class MergerShockInputs:
    """Read-only, memory-mapped detected cells from saved ShockFinder outputs.

    Construct with cache_merger_shock_inputs/load_merger_shock_inputs. data
    holds ALL detections and their existing validity mask, including invalid
    detections; geometry is physical kpc with unit normals. This is not a dense
    ShockResult: use get_merger_shock_members for original center-ID lookup.
    """
    snapshot: int
    data: dict
    extras: dict
    directory: str


def cache_merger_shock_inputs(iout, result, dissipation, directory, *,
                             chunk_size=131072, include_endpoints=True, provenance=None):
    """Save detected-cell arrays once, then release the huge original inputs.

    This does NOT run ShockFinder. It preserves existing IDs, validity decisions
    and dissipation; applies only the same explicit physical-kpc/normal
    conversion used by merger_shock_catalog. Arrays are written directly to
    disk-backed .npy files in bounded chunks. include_endpoints also retains
    saved upstream/downstream positions for later galaxy encounter geometry.

    A fresh directory is required; existing files are never overwritten.
    metadata.json is written last, so an interrupted cache cannot be loaded as
    complete. provenance may record original file paths/checksums. Original
    pickles are never modified. The returned cache can be passed as result,
    with dissipation=None. Subsequent runs need only load_merger_shock_inputs.
    """
    if isinstance(iout, bool) or not isinstance(iout, (int, np.integer)):
        raise ValueError("iout must be an integer snapshot number")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(result, MergerShockInputs):
        raise ValueError("inputs are already cached; load or reuse that cache")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"cache directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    data = _merger_compact_results(result, dissipation, snapshot=int(iout), chunk_size=chunk_size, storage_dir=directory)
    n, count = data["retained_count"], len(data["retained_row"])
    arrays = {name: {"shape": list(value.shape), "dtype": value.dtype.str} for name, value in data.items() if isinstance(value, np.ndarray)}
    extra_names = []
    for name in ("center_index", "upstream_index", "downstream_index", "level", "zone_width", "mach_consistent", "mach_validation_status"):
        if _merger_value(result, name) is None:
            continue
        source = _merger_array(result, name, n)
        if source.dtype.hasobject:
            raise ValueError(f"cache field {name} must be a numeric array")
        saved = np.lib.format.open_memmap(directory/f"extra_{name}.npy", mode="w+", dtype=source.dtype, shape=(count,)) if count else np.empty(0, dtype=source.dtype)
        for start in range(0, count, chunk_size):
            target = slice(start, start+chunk_size)
            saved[target] = source[data["retained_row"][target]]
        if isinstance(saved, np.memmap):
            saved.flush()
        else:
            np.save(directory/f"extra_{name}.npy", saved, allow_pickle=False)
        arrays[f"extra_{name}"] = {"shape": list(saved.shape), "dtype": saved.dtype.str}
        extra_names.append(name)
        del saved
    if include_endpoints:
        factor = _MERGER_LENGTH_TO_KPC[data["result_position_unit"]]
        positions = _merger_array(result, "pos", shape=(n, 3))
        for prefix in ("upstream", "downstream"):
            if _merger_value(result, prefix+"_index") is None:
                continue
            name = prefix+"_pos"
            indices = np.load(directory/f"extra_{prefix}_index.npy", mmap_mode="r", allow_pickle=False) if count else np.empty(0, dtype=int)
            saved = np.lib.format.open_memmap(directory/f"extra_{name}.npy", mode="w+", dtype=np.float64, shape=(count, 3)) if count else np.empty((0, 3))
            for start in range(0, count, chunk_size):
                target = slice(start, start+chunk_size)
                ids = indices[target]
                valid = (ids >= 0) & (ids < n)
                saved[target] = np.nan
                block = saved[target]
                block[valid] = positions[ids[valid]]*factor
            if isinstance(saved, np.memmap):
                saved.flush()
            else:
                np.save(directory/f"extra_{name}.npy", saved, allow_pickle=False)
            arrays[f"extra_{name}"] = {"shape": list(saved.shape), "dtype": saved.dtype.str}
            extra_names.append(name)
            del saved, indices
    for value in data.values():
        if isinstance(value, np.memmap):
            value.flush()
    metadata = {"cache_schema_version": 1, "snapshot": int(iout), "arrays": arrays, "extras": extra_names,
                "scalars": {name: value for name, value in data.items() if not isinstance(value, np.ndarray)},
                "geometry_unit": "physical kpc", "normals": "unit vectors", "provenance": provenance}
    temporary = directory/"metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory/"metadata.json")
    return load_merger_shock_inputs(directory)


def load_merger_shock_inputs(directory):
    """Open a completed cache using read-only NumPy memory maps, no pickle I/O."""
    directory = Path(directory).resolve()
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("cache_schema_version") != 1 or metadata.get("geometry_unit") != "physical kpc"
            or metadata.get("normals") != "unit vectors"):
        raise ValueError("unsupported merger-input cache schema or units")
    allowed = {"retained_row", "cell_id", "pos", "normal", "valid", "mach", "dx", "flux", "total", "area"}
    extras_allowed = {"center_index", "upstream_index", "downstream_index", "level", "zone_width", "mach_consistent",
                      "mach_validation_status", "upstream_pos", "downstream_pos"}
    if set(metadata["extras"])-extras_allowed or set(metadata["arrays"])-(allowed | {"extra_"+k for k in extras_allowed}):
        raise ValueError("unexpected cache fields")
    data, extras = dict(metadata["scalars"]), {}
    n = data.get("retained_count")
    if (isinstance(n, bool) or not isinstance(n, int) or n < 0
            or data.get("result_position_unit") not in _MERGER_LENGTH_TO_KPC
            or not isinstance(data.get("validation_unknown"), bool)
            or data.get("snapshot") != metadata["snapshot"]):
        raise ValueError("invalid cache source metadata")
    for name, description in metadata["arrays"].items():
        path = directory/f"{name}.npy"
        # mmap cannot map an empty file's data region; empty arrays are tiny.
        mode = "r" if np.prod(description["shape"], dtype=np.int64) else None
        array = np.load(path, mmap_mode=mode, allow_pickle=False)
        if list(array.shape) != description["shape"] or array.dtype.str != description["dtype"]:
            raise ValueError(f"cache array {name} disagrees with metadata")
        array.flags.writeable = False
        (extras if name.startswith("extra_") else data)[name[6:] if name.startswith("extra_") else name] = array
    if allowed-set(data) or set(metadata["extras"]) != set(extras):
        raise ValueError("incomplete merger-input cache")
    count = len(data["retained_row"])
    for name in allowed:
        expected = (count, 3) if name in ("pos", "normal") else (count,)
        if data[name].shape != expected:
            raise ValueError(f"cache array {name} has incompatible shape")
    if (data["retained_row"].dtype.kind not in "iu" or data["cell_id"].dtype.kind not in "iu"
            or data["valid"].dtype.kind != "b"):
        raise ValueError("cache IDs/validity need integer/boolean dtypes")
    for name, array in extras.items():
        expected = (count, 3) if name.endswith("_pos") else (count,)
        if array.shape != expected:
            raise ValueError(f"cache array extra_{name} has incompatible shape")
    # Member lookup uses original dense IDs in sorted order. Check them in
    # bounded slices rather than allocating a full-size diff array.
    rows = data["retained_row"]
    for start in range(0, count, 131072):
        block = rows[max(0, start-1):start+131072]
        if (np.any(block < 0) or np.any(block >= n) or np.any(block[1:] <= block[:-1])):
            raise ValueError("cache center IDs must increase within the original retained-row range")
    return MergerShockInputs(int(metadata["snapshot"]), data, extras, str(directory))


def _merger_value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _merger_required(obj, name):
    value = _merger_value(obj, name)
    if value is None:
        raise ValueError(f"saved input lacks required field {name!r}")
    return value


def _merger_minimum_image(vec, box_size):
    if box_size is None:
        return vec
    return vec - box_size * np.round(vec / box_size)


def _merger_geometry(centers, config):
    c1, c2, _ = centers
    if c2 is None:
        return {"center1": c1, "center2": None, "midpoint": None,
                "origin": c1.copy(), "propagation_origin_type": "primary_remnant",
                "separation_vector": None, "separation_kpc": math.nan,
                "axis": np.full(3, math.nan), "axis_source": "unavailable",
                "secondary_center_available": False, "geometry_source": "primary_only_after_secondary_termination",
                "quality_flags": ["secondary_center_unavailable", "separation_unavailable", "remnant_origin_used"]}
    sep = _merger_minimum_image(c2 - c1, config.box_size_kpc)
    distance = float(np.linalg.norm(sep))
    midpoint = c1 + 0.5 * sep
    return {"center1": c1, "center2": c2, "midpoint": midpoint,
            "origin": midpoint.copy(), "propagation_origin_type": "two_center_midpoint",
            "separation_vector": sep, "separation_kpc": distance,
            "axis": sep / distance if distance > 0 else np.full(3, math.nan),
            "axis_source": "instantaneous_separation" if distance > 0 else "unavailable",
            "secondary_center_available": True, "geometry_source": "two_tracked_centers",
            "quality_flags": [] if distance > 0 else ["merger_axis_unavailable"]}


def _merger_geometry_history(centers, meta, history, config):
    """Use measured separation while available, then an explicit remnant frame.

    Holding the last measured axis is a configurable geometrical convention,
    not an extrapolation of the secondary trajectory or separation.
    """
    result = {}
    last_axis = None
    for snap in history:
        geom = _merger_geometry(centers[snap], config)
        if geom["secondary_center_available"] and np.all(np.isfinite(geom["axis"])):
            last_axis = (snap, meta[snap]["time_gyr"], geom["axis"].copy())
            reference = last_axis
        elif not geom["secondary_center_available"] and last_axis is not None and config.post_merger_axis_policy == "last_measured":
            geom["axis"] = last_axis[2].copy()
            geom["axis_source"] = "last_measured_two_center_axis"
            geom["quality_flags"].append("axis_carried_forward")
            reference = last_axis
        else:
            reference = None
            if "merger_axis_unavailable" not in geom["quality_flags"]:
                geom["quality_flags"].append("merger_axis_unavailable")
        geom["axis_reference_snapshot"] = reference[0] if reference is not None else None
        geom["axis_reference_time_gyr"] = reference[1] if reference is not None else None
        geom["axis_age_gyr"] = meta[snap]["time_gyr"] - reference[1] if reference is not None else math.nan
        result[snap] = geom
    return result


def _merger_verify_epochs(config, centers, meta, history):
    paired = [snap for snap in history if centers[snap][1] is not None]
    distances = np.array([_merger_geometry(centers[s], config)["separation_kpc"] for s in paired])
    complete_radii = lambda snap: centers[snap][2] is not None and all(r is not None for r in centers[snap][2])
    overlap_candidates = [i for i, snap in enumerate(paired) if complete_radii(snap) and distances[i] <= sum(centers[snap][2])]
    overlap = overlap_candidates[0] if overlap_candidates else None
    start = overlap if overlap is not None else 0
    minima = [i for i in range(max(1, start), len(paired) - 1) if distances[i] <= distances[i-1] and distances[i] < distances[i+1]]
    peri = minima[0] if minima else int(np.argmin(distances[start:]) + start) if len(paired) else None
    maxima = [i for i in range(peri + 1, len(paired)-1) if distances[i] >= distances[i-1] and distances[i] > distances[i+1]] if peri is not None else []
    apo = maxima[0] if maxima else None
    found = {"overlap": overlap, "pericenter": peri, "apocenter": apo}
    report = {"history_snapshots": len(history), "two_center_history_snapshots": len(paired),
              "last_observed_secondary_snapshot": paired[-1] if paired else None,
              "first_primary_only_snapshot": next((s for s in history if centers[s][1] is None), None),
              "epoch_geometry": "measured two-center separations only; terminal branch loss is not zero separation",
              "separation_history": []}
    for snap in history:
        geom = _merger_geometry(centers[snap], config)
        report["separation_history"].append({"snapshot": snap, "time_gyr": meta[snap]["time_gyr"],
            "redshift": meta[snap]["redshift"], "secondary_center_available": geom["secondary_center_available"],
            "separation_kpc": geom["separation_kpc"] if geom["secondary_center_available"] else None,
            "separation_vector_kpc": geom["separation_vector"].tolist() if geom["separation_vector"] is not None else None,
            "instantaneous_axis": geom["axis"].tolist() if np.all(np.isfinite(geom["axis"])) else None})
    for name, index in found.items():
        ref = config.reference_epochs[name]
        if index is None:
            reasons = {"overlap": "requires two centers and virial-radius crossing",
                       "pericenter": "no measured two-center separation",
                       "apocenter": "requires a post-passage two-center local maximum"}
            report[name] = {"verified": False, "reason": reasons[name]}
            continue
        snap = paired[index]
        delta_snap = abs(snap - ref["snapshot"])
        delta_z = abs(meta[snap]["redshift"] - ref["redshift"])
        bracketed_overlap = (name != "overlap" or
            (index > 0 and complete_radii(paired[index - 1]) and
             distances[index - 1] > sum(centers[paired[index - 1]][2])))
        extremum_bracketed = name != "pericenter" or bool(minima)
        report[name] = {"snapshot": snap, "time_gyr": meta[snap]["time_gyr"],
            "redshift": meta[snap]["redshift"], "separation_kpc": distances[index],
            "verified": bool(bracketed_overlap and extremum_bracketed and delta_snap <= config.reference_snapshot_tolerance and delta_z <= config.reference_redshift_tolerance),
            "reference_snapshot_delta": delta_snap, "reference_redshift_delta": delta_z}
        if name == "overlap":
            report[name]["crossing_bracketed_by_outputs"] = bool(bracketed_overlap)
        else:
            report[name]["local_extremum_bracketed_by_outputs"] = bool(extremum_bracketed)
        if name == "pericenter" and not extremum_bracketed:
            report[name]["reason"] = "sampled minimum has no resolved turning point; branch termination cannot verify core passage"
    return report


def _merger_array(obj, name, n=None, shape=None, dtype=None):
    arr = np.asarray(_merger_required(obj, name), dtype=dtype)
    expected = shape if shape is not None else (n,)
    if arr.shape != expected:
        raise ValueError(f"{name} shape {arr.shape}; expected {expected}")
    return arr


def _merger_compact_results(result, dissipation, *, snapshot=None, chunk_size=131072, storage_dir=None):
    """Read selected cells in bounded chunks; never copy a full dense field."""
    if isinstance(result, MergerShockInputs):
        if dissipation is not None:
            raise ValueError("cached inputs already include saved dissipation; pass None")
        if snapshot is not None and result.snapshot != snapshot:
            raise ValueError("cache snapshot does not match iout")
        return result.data
    n = len(_merger_required(result, "mach"))
    unit = _merger_required(result, "position_unit")
    if unit not in _MERGER_LENGTH_TO_KPC:
        raise ValueError(f"unsupported or unspecified result.position_unit {unit!r}")
    if dissipation is None:
        raise ValueError("no saved dissipation supplied; do not recompute it")
    factor = _MERGER_LENGTH_TO_KPC[unit]
    shock = _merger_array(result, "shock", n)
    count = int(np.count_nonzero(shock))
    index_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    source = {key: _merger_array(result, key, n) for key in ("mach", "dx", "selected_indices", "center_index")}
    source["pos"] = _merger_array(result, "pos", shape=(n, 3))
    source["normal"] = _merger_array(result, "normal", shape=(n, 3))
    optional = {key: _merger_array(result, key, n) for key in ("mach_consistent", "mach_validation_status", "upstream_index", "downstream_index")
                if _merger_value(result, key) is not None}
    diss = {key: _merger_array(dissipation, key, n) for key in ("flux", "total", "area")}
    def allocate(name, shape, dtype=np.float64):
        if storage_dir is None:
            return np.empty(shape, dtype=dtype)
        if not count:
            array = np.empty(shape, dtype=dtype)
            np.save(Path(storage_dir)/f"{name}.npy", array, allow_pickle=False)
            return array
        return np.lib.format.open_memmap(Path(storage_dir)/f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    data = {"retained_count": n, "result_position_unit": unit, "snapshot": snapshot,
            "retained_row": allocate("retained_row", (count,), index_dtype), "cell_id": allocate("cell_id", (count,), np.int64),
            "pos": allocate("pos", (count, 3)), "normal": allocate("normal", (count, 3)), "valid": allocate("valid", (count,), bool),
            "validation_unknown": "mach_consistent" not in optional and "mach_validation_status" not in optional}
    for key in ("mach", "dx", "flux", "total", "area"):
        data[key] = allocate(key, (count,))
    offset = 0
    # Scan the mask in chunks too: flatnonzero on a huge dense output otherwise
    # allocates int64 IDs for every detection before any field is compacted.
    for start in range(0, n, chunk_size):
        rows = np.flatnonzero(shock[start:start+chunk_size])+start
        if not len(rows):
            continue
        stop = offset+len(rows)
        target = slice(offset, stop)
        center = source["center_index"][rows]
        if center.dtype.kind not in "iu" or not np.array_equal(center, rows):
            raise ValueError("expected dense ShockResult: accepted center_index must equal its retained result row")
        data["retained_row"][target] = rows
        data["cell_id"][target] = source["selected_indices"][rows]
        for key in ("mach", "dx", "pos", "normal"):
            data[key][target] = source[key][rows]
        data["pos"][target] *= factor
        data["dx"][target] *= factor
        normal = data["normal"][target]
        norm = np.linalg.norm(normal, axis=1)
        mach, pos, dx = data["mach"][target], data["pos"][target], data["dx"][target]
        valid = (np.isfinite(mach) & (mach > 1) & np.all(np.isfinite(pos), axis=1)
                 & np.isfinite(dx) & (dx > 0) & np.isfinite(norm) & (norm > 0)
                 & np.all(np.isfinite(normal), axis=1))
        normal /= np.maximum(norm[:, None], 1.e-300)
        if "mach_consistent" in optional:
            valid &= optional["mach_consistent"][rows].astype(bool)
        if "mach_validation_status" in optional:
            status = optional["mach_validation_status"][rows].astype(np.int64)
            valid &= (status & (1 << 8)) == 0  # saved ENDPOINT_INVALID
            if "mach_consistent" not in optional:
                valid &= (status & (1 << 7)) != 0  # saved MACH_CONSISTENT
        if "upstream_index" in optional and "downstream_index" in optional:
            valid &= (optional["upstream_index"][rows] >= 0) & (optional["downstream_index"][rows] >= 0)
        for key in ("flux", "total", "area"):
            data[key][target] = diss[key][rows]
            valid &= np.isfinite(data[key][target]) & (data[key][target] > 0 if key == "area" else data[key][target] >= 0)
        data["valid"][target] = valid
        offset = stop
    ids = data["cell_id"]
    increasing = all(np.all(ids[max(0, start-1):min(count, start+chunk_size)][1:]
                            > ids[max(0, start-1):min(count, start+chunk_size)][:-1])
                     for start in range(0, count, chunk_size))
    if not increasing and len(np.unique(ids)) != count:
        raise ValueError("shock cell identifiers are not unique within snapshot")
    return data


class _MergerComponents:
    """Merge bounded edge batches in SciPy, without retaining the full graph."""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32 if n <= np.iinfo(np.int32).max else np.int64)

    def roots(self, indices):
        roots = self.parent[indices]
        while True:
            next_roots = self.parent[roots]
            if np.array_equal(roots, next_roots):
                break
            roots = next_roots
        self.parent[indices] = roots
        return roots

    def merge(self, left, right):
        if not len(left):
            return
        left, right = self.roots(left), self.roots(right)
        use = left != right
        if not np.any(use):
            return
        left, right = left[use], right[use]
        # Compress to roots touched by this batch rather than allocating a
        # graph with ALL shock cells for each chunk. Connections from earlier
        # batches are represented by their roots, and therefore never lost.
        unique, inverse = np.unique(np.concatenate((left, right)), return_inverse=True)
        m = len(left)
        graph = coo_matrix((np.ones(m, dtype=bool), (inverse[:m], inverse[m:])),
                           shape=(len(unique), len(unique))).tocsr()
        count, labels = connected_components(graph, directed=False)
        minimum = np.full(count, np.iinfo(self.parent.dtype).max, dtype=self.parent.dtype)
        np.minimum.at(minimum, labels, unique)
        self.parent[unique] = minimum[labels]


def _merger_neighbor_backend(data, config):
    available = _merger_neighbors is not None and len(data["valid"]) <= np.iinfo(np.int32).max
    if config.neighbor_backend == "fortran" and not available:
        raise RuntimeError("Fortran merger neighbors require the separate _merger_neighbors extension and int32-sized compact inputs. "
                           "Build it from shocktest/: python -m numpy.f2py -c fortran/merger_neighbors.f90 -m _merger_neighbors "
                           "--f90flags='-O3 -ffp-contract=off'; or select neighbor_backend='scipy'.")
    return "fortran" if available and config.neighbor_backend != "scipy" else "scipy"


class _MergerFortranComponents(_MergerComponents):
    """Borrow canonical geometry and union pairs in the optional compiled kernel."""
    def accept(self, left, right, rows, data, config, same_bucket):
        _merger_neighbors.merger_neighbor_kernel.merge_neighbor_pairs(
            data["pos"].T, data["dx"], data["normal"].T, rows, left, right, self.parent,
            config.max_cell_gap_factor, config.minimum_neighbor_normal_cosine,
            config.box_size_kpc or 0., int(same_bucket))


def _merger_accept_edges(left, right, rows, data, dx, components, config, same_bucket):
    if isinstance(components, _MergerFortranComponents):
        components.accept(left, right, rows, data, config, same_bucket)
        return
    if same_bucket:
        use = right > left
        left, right = left[use], right[use]
    if not len(left):
        return
    reach = .5*(dx[left]+dx[right])+config.max_cell_gap_factor*np.maximum(dx[left], dx[right])
    a, b = rows[left], rows[right]
    delta = _merger_minimum_image(data["pos"][a]-data["pos"][b], config.box_size_kpc)
    close = np.all(np.abs(delta) <= reach[:, None], axis=1)
    if not np.any(close):
        return
    left, right, a, b = left[close], right[close], a[close], b[close]
    alignment = np.abs(np.einsum("ij,ij->i", data["normal"][a], data["normal"][b]))
    accepted = alignment >= config.minimum_neighbor_normal_cosine
    components.merge(left[accepted], right[accepted])


def _merger_group_cells(data, config):
    """Exact AMR cube contact + normal cut, with bounded neighbor workspace."""
    valid = data["valid"]
    index_dtype = np.int32 if len(valid) <= np.iinfo(np.int32).max else np.int64
    all_valid = bool(np.all(valid))
    rows = np.arange(len(valid), dtype=index_dtype) if all_valid else np.flatnonzero(valid).astype(index_dtype)
    if len(rows) == 0:
        return []
    dx = data["dx"] if all_valid else data["dx"][rows]
    backend = _merger_neighbor_backend(data, config)
    components = (_MergerFortranComponents if backend == "fortran" else _MergerComponents)(len(rows))
    # AMR-scale trees keep fine-cell searches local even when a few coarse
    # cells are present in the same snapshot.
    levels = np.empty(len(rows), dtype=np.int32)
    minimum_dx = np.min(dx)
    for start in range(0, len(rows), config.cell_chunk_size):
        chunk = slice(start, start+config.cell_chunk_size)
        levels[chunk] = np.floor(np.log2(dx[chunk]/minimum_dx)+1.e-8)
    buckets = []
    for level in np.unique(levels):
        indices = np.flatnonzero(levels == level).astype(index_dtype)
        buckets.append([indices, None, float(np.max(dx[indices]))])
    del levels
    for a, (source_indices, _, _) in enumerate(buckets):
        for b in range(a, len(buckets)):
            target_indices, tree, target_max_dx = buckets[b]
            if backend == "fortran":
                status = _merger_neighbors.merger_neighbor_kernel.connect_bucket(
                    data["pos"].T, data["dx"], data["normal"].T, rows,
                    source_indices, target_indices, components.parent,
                    config.max_cell_gap_factor, config.minimum_neighbor_normal_cosine,
                    config.box_size_kpc or 0., int(a == b))
                if status == 0:
                    continue
            if tree is None:
                if all_valid and len(target_indices) == len(rows) and config.box_size_kpc is None:
                    coordinates = data["pos"]  # cKDTree borrows contiguous float64.
                else:
                    coordinates = data["pos"][rows[target_indices]]
                    if config.box_size_kpc is not None:
                        np.remainder(coordinates, config.box_size_kpc, out=coordinates)
                tree = cKDTree(coordinates, boxsize=config.box_size_kpc, copy_data=False)
                buckets[b][1] = tree
            for start in range(0, len(source_indices), config.spatial_query_chunk):
                chunk = source_indices[start:start + config.spatial_query_chunk]
                points = data["pos"][rows[chunk]]
                if config.box_size_kpc is not None:
                    points %= config.box_size_kpc
                # Chebyshev queries fit the axis-aligned AMR contact criterion:
                # the old sqrt(3) spherical search returned extra neighbors.
                radii = (0.5 * (dx[chunk] + target_max_dx) +
                    config.max_cell_gap_factor * np.maximum(dx[chunk], target_max_dx))
                counts = tree.query_ball_point(points, radii, p=np.inf, return_length=True)
                cumulative = np.r_[0, np.cumsum(counts)]
                lo = 0
                while lo < len(chunk):
                    if counts[lo] > config.max_neighbor_pairs:
                        # Even ONE unusually dense neighborhood must not create
                        # an unbounded Python list. Scan its target bucket in
                        # bounded blocks with the exact same acceptance cuts.
                        for k in range(0, len(target_indices), config.max_neighbor_pairs):
                            right = target_indices[k:k+config.max_neighbor_pairs]
                            left = np.full(len(right), chunk[lo], dtype=index_dtype)
                            _merger_accept_edges(left, right, rows, data, dx, components, config, a == b)
                        lo += 1
                        continue
                    hi = int(np.searchsorted(cumulative, cumulative[lo]+config.max_neighbor_pairs, side="right")-1)
                    hi = min(max(lo+1, hi), len(chunk))
                    neighborhoods = tree.query_ball_point(points[lo:hi], radii[lo:hi], p=np.inf)
                    total = int(np.sum(counts[lo:hi]))
                    if total:
                        local = np.concatenate(neighborhoods).astype(index_dtype, copy=False)
                        left = np.repeat(chunk[lo:hi], counts[lo:hi])
                        right = target_indices[local]
                        _merger_accept_edges(left, right, rows, data, dx, components, config, a == b)
                    lo = hi
    del buckets
    roots = components.roots(np.arange(len(rows), dtype=index_dtype))
    order = np.argsort(roots, kind="stable")
    cuts = np.flatnonzero(np.diff(roots[order])) + 1
    return [rows[group] for group in np.split(order, cuts)]


def _merger_summarize_front(rows, data, geom, snapshot, time_gyr, redshift, config):
    area = data["area"][rows]
    weights = area / area.sum()
    center, normal_sum = np.zeros(3), np.zeros(3)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    reference = data["normal"][rows[np.argmax(area)]]
    anchor = data["pos"][rows[0]]
    unwrapped = False
    for start in range(0, len(rows), config.cell_chunk_size):
        chunk = slice(start, start+config.cell_chunk_size)
        take = rows[chunk]
        pos, dx = data["pos"][take], data["dx"][take]
        if config.box_size_kpc is not None:
            local = anchor+_merger_minimum_image(pos-anchor, config.box_size_kpc)
            unwrapped |= bool(np.any(local != pos))
            pos = local
        center += np.sum(pos*weights[chunk, None], axis=0)
        lower = np.minimum(lower, np.min(pos-dx[:, None]/2, axis=0))
        upper = np.maximum(upper, np.max(pos+dx[:, None]/2, axis=0))
        normals = data["normal"][take]
        normals *= np.where(normals@reference < 0, -1., 1.)[:, None]
        normal_sum += np.sum(normals*area[chunk, None], axis=0)
    extent = upper - lower
    mean = normal_sum/area.sum()
    coherence = float(np.linalg.norm(mean))
    normal = mean/max(coherence, 1.e-300)
    origin_vector = _merger_minimum_image(center - geom["origin"], config.box_size_kpc)
    axis_available = np.all(np.isfinite(geom["axis"]))
    axis_coordinate = float(origin_vector @ geom["axis"]) if axis_available else math.nan
    axis_offset = float(np.linalg.norm(origin_vector - axis_coordinate * geom["axis"])) if axis_available else math.nan
    distances = [float(np.linalg.norm(_merger_minimum_image(center - geom[f"center{i}"], config.box_size_kpc)))
                 if geom[f"center{i}"] is not None else math.nan for i in (1, 2)]
    second_center = geom["center2"] if geom["center2"] is not None else np.full(3, math.nan)
    mach = data["mach"][rows]
    mach_min, mach_max = float(np.min(mach)), float(np.max(mach))
    mach_p10, mach_median, mach_p90 = np.quantile(mach, (.1, .5, .9), overwrite_input=True)
    total = data["total"][rows]
    flux = data["flux"][rows]
    flags = list(geom["quality_flags"])
    if len(rows) < config.minimum_front_cells:
        flags.append("few_cells")
    if data["validation_unknown"]:
        flags.append("validation_unavailable")
    if coherence < config.minimum_neighbor_normal_cosine:
        flags.append("low_normal_coherence")
    if unwrapped:
        flags.append("periodic_front_unwrapped")
    reference_sign = np.sign(data["normal"][rows[0]]@normal)
    for start in range(0, len(rows), config.cell_chunk_size):
        if np.any(np.sign(data["normal"][rows[start:start+config.cell_chunk_size]]@normal) != reference_sign):
            flags.append("normal_signs_mixed")
            break
    total_sum = float(np.sum(total))
    total_median = float(np.median(total, overwrite_input=True))
    flux_median = float(np.median(flux, overwrite_input=True))
    return {
        "snapshot": snapshot, "time_gyr": time_gyr, "redshift": redshift,
        "front_id": "", "track_id": "", "previous_front_id": "", "next_front_id": "",
        "cell_count": int(len(rows)), "rows": rows,
        "center_x_kpc": float(center[0]), "center_y_kpc": float(center[1]), "center_z_kpc": float(center[2]),
        "xmin_kpc": float(lower[0]), "xmax_kpc": float(upper[0]),
        "ymin_kpc": float(lower[1]), "ymax_kpc": float(upper[1]),
        "zmin_kpc": float(lower[2]), "zmax_kpc": float(upper[2]),
        "extent_x_kpc": float(extent[0]), "extent_y_kpc": float(extent[1]), "extent_z_kpc": float(extent[2]),
        "max_extent_kpc": float(np.max(extent)), "area_kpc2": float(np.sum(area)),
        "distance_center1_kpc": distances[0], "distance_center2_kpc": distances[1],
        "center1_x_kpc": float(geom["center1"][0]), "center1_y_kpc": float(geom["center1"][1]), "center1_z_kpc": float(geom["center1"][2]),
        "center2_x_kpc": float(second_center[0]), "center2_y_kpc": float(second_center[1]), "center2_z_kpc": float(second_center[2]),
        "secondary_center_available": geom["secondary_center_available"],
        "geometry_source": geom["geometry_source"], "axis_source": geom["axis_source"],
        "axis_reference_snapshot": geom.get("axis_reference_snapshot"),
        "axis_reference_time_gyr": geom.get("axis_reference_time_gyr"),
        "axis_age_gyr": geom.get("axis_age_gyr", math.nan),
        "propagation_origin_type": geom["propagation_origin_type"],
        "origin_x_kpc": float(geom["origin"][0]), "origin_y_kpc": float(geom["origin"][1]), "origin_z_kpc": float(geom["origin"][2]),
        "distance_origin_kpc": float(np.linalg.norm(origin_vector)),
        "distance_midpoint_kpc": float(np.linalg.norm(origin_vector)) if geom["midpoint"] is not None else math.nan,
        "axis_coordinate_kpc": axis_coordinate, "axis_offset_kpc": axis_offset,
        "separation_kpc": geom["separation_kpc"],
        "axis_x": float(geom["axis"][0]), "axis_y": float(geom["axis"][1]), "axis_z": float(geom["axis"][2]),
        "normal_x": float(normal[0]), "normal_y": float(normal[1]), "normal_z": float(normal[2]),
        "normal_coherence": coherence,
        "normal_axis_angle_deg": float(np.degrees(np.arccos(np.clip(abs(normal @ geom["axis"]), 0, 1)))) if axis_available else math.nan,
        "mach_min": mach_min, "mach_p10": float(mach_p10),
        "mach_median": float(mach_median), "mach_p90": float(mach_p90),
        "mach_max": mach_max,
        "dissipation_total_erg_s": total_sum,
        "dissipation_median_erg_s": total_median,
        "dissipation_flux_median_erg_s_kpc2": flux_median,
        "propagation_x": math.nan, "propagation_y": math.nan, "propagation_z": math.nan,
        "propagation_speed_kpc_gyr": math.nan, "propagation_speed_km_s": math.nan,
        "outward_axis_speed_kpc_gyr": math.nan, "outward_radial_speed_kpc_gyr": math.nan,
        "merger_evidence_score": math.nan, "confidence": "", "classification": "",
        "uncertain": True, "quality_flags": flags,
    }


def _merger_front_center(front):
    return np.array([front[f"center_{a}_kpc"] for a in "xyz"])


def _merger_front_normal(front):
    return np.array([front[f"normal_{a}"] for a in "xyz"])


def _merger_front_origin(front):
    return np.array([front[f"origin_{a}_kpc"] for a in "xyz"])


def _merger_front_axis(front):
    return np.array([front[f"axis_{a}"] for a in "xyz"])


def _merger_link_fronts(by_snapshot, entries, meta, config):
    previous = []
    previous_time = None
    for entry in entries:
        snap = entry["snapshot"]
        current = by_snapshot[snap]
        for front in current:
            front["front_id"] = f"{snap}:{front['representative_shock_id']}"
        dt = None if previous_time is None else meta[snap]["time_gyr"] - previous_time
        matches = []
        if previous and current and dt is not None and 0 < dt <= config.max_link_interval_gyr:
            old_centers = np.array([_merger_front_center(front) for front in previous])
            new_centers = np.array([_merger_front_center(front) for front in current])
            if config.box_size_kpc is None:
                tree = cKDTree(new_centers)
                query_centers = old_centers
            else:
                tree = cKDTree(new_centers % config.box_size_kpc, boxsize=config.box_size_kpc)
                query_centers = old_centers % config.box_size_kpc
            radii = config.max_front_speed_kpc_gyr * dt + 0.5 * (
                np.array([front["max_extent_kpc"] for front in previous]) +
                max(front["max_extent_kpc"] for front in current))
            neighborhoods = tree.query_ball_point(query_centers, radii)
            edges = []
            old_degree = np.zeros(len(previous), dtype=np.int32)
            new_degree = np.zeros(len(current), dtype=np.int32)
            for i, old in enumerate(previous):
                for j in neighborhoods[i]:
                    new = current[j]
                    distance = np.linalg.norm(_merger_minimum_image(_merger_front_center(new) - _merger_front_center(old), config.box_size_kpc))
                    allowance = config.max_front_speed_kpc_gyr * dt + 0.5 * (old["max_extent_kpc"] + new["max_extent_kpc"])
                    normal_cos = abs(float(_merger_front_normal(old) @ _merger_front_normal(new)))
                    same_origin_type = old["propagation_origin_type"] == new["propagation_origin_type"]
                    axis = _merger_front_axis(new)
                    branch_change = False
                    if same_origin_type and np.all(np.isfinite(axis)):
                        # Compare both fronts in one axis frame; rotation of
                        # the instantaneous separation axis is not propagation.
                        old_coordinate = _merger_minimum_image(_merger_front_center(old) - _merger_front_origin(old), config.box_size_kpc) @ axis
                        new_coordinate = _merger_minimum_image(_merger_front_center(new) - _merger_front_origin(new), config.box_size_kpc) @ axis
                        branch_change = old_coordinate * new_coordinate < 0
                    if distance <= allowance and normal_cos >= config.minimum_neighbor_normal_cosine and not branch_change:
                        cost = distance / allowance + 0.5 * (1 - normal_cos) + 0.1 * abs(math.log(new["mach_median"] / old["mach_median"]))
                        edges.append((cost, i, j))
                        old_degree[i] += 1
                        new_degree[j] += 1
            used_old, used_new = set(), set()
            for _, i, j in sorted(edges):
                if i not in used_old and j not in used_new:
                    matches.append((i, j))
                    used_old.add(i)
                    used_new.add(j)
            for i, j in matches:
                if old_degree[i] > 1 or new_degree[j] > 1:
                    previous[i]["quality_flags"].append("ambiguous_temporal_link")
                    current[j]["quality_flags"].append("ambiguous_temporal_link")
        linked_new = set()
        for i, j in matches:
            old, new = previous[i], current[j]
            new["track_id"] = old["track_id"]
            new["previous_front_id"] = old["front_id"]
            old["next_front_id"] = new["front_id"]
            displacement = _merger_minimum_image(_merger_front_center(new) - _merger_front_center(old), config.box_size_kpc)
            speed = float(np.linalg.norm(displacement) / dt)
            new["propagation_speed_kpc_gyr"] = speed
            new["propagation_speed_km_s"] = speed * _MERGER_KPC_PER_GYR_TO_KMS
            if speed > 0:
                direction = displacement / np.linalg.norm(displacement)
                for axis, value in zip("xyz", direction):
                    new[f"propagation_{axis}"] = float(value)
            if old["propagation_origin_type"] == new["propagation_origin_type"]:
                origin_displacement = _merger_minimum_image(_merger_front_origin(new) - _merger_front_origin(old), config.box_size_kpc)
                relative_displacement = displacement - origin_displacement
                axis = _merger_front_axis(new)
                if np.all(np.isfinite(axis)):
                    old_coordinate = _merger_minimum_image(_merger_front_center(old) - _merger_front_origin(old), config.box_size_kpc) @ axis
                    new_coordinate = _merger_minimum_image(_merger_front_center(new) - _merger_front_origin(new), config.box_size_kpc) @ axis
                    branch = np.sign(new_coordinate) if new_coordinate != 0 else np.sign(old_coordinate)
                    new["outward_axis_speed_kpc_gyr"] = float(branch * (relative_displacement @ axis) / dt)
                new["outward_radial_speed_kpc_gyr"] = float((new["distance_origin_kpc"] - old["distance_origin_kpc"]) / dt)
            else:
                # Keep the link and absolute velocity, but the midpoint-to-
                # remnant origin change cannot supply an outward velocity.
                new["quality_flags"].append("propagation_origin_transition")
            linked_new.add(j)
        for j, front in enumerate(current):
            if j not in linked_new:
                front["track_id"] = front["front_id"]
        previous, previous_time = current, meta[snap]["time_gyr"]


def _merger_score_fronts(by_snapshot, epochs, config):
    fronts = [front for group in by_snapshot.values() for front in group]
    if not fronts:
        return None
    counts = {}
    births = {}
    motions = {}
    for front in fronts:
        counts[front["track_id"]] = counts.get(front["track_id"], 0) + 1
        if front["track_id"] not in births or front["time_gyr"] < births[front["track_id"]]["time_gyr"]:
            births[front["track_id"]] = front
        if math.isfinite(front["outward_axis_speed_kpc_gyr"]):
            front["outward_motion_basis"] = "axis"
            motions.setdefault(front["track_id"], []).append(front["outward_axis_speed_kpc_gyr"])
        elif math.isfinite(front["outward_radial_speed_kpc_gyr"]):
            front["outward_motion_basis"] = "radial"
            motions.setdefault(front["track_id"], []).append(front["outward_radial_speed_kpc_gyr"])
        else:
            front["outward_motion_basis"] = "unmeasured"
    positive_diss = [f["dissipation_total_erg_s"] for f in fronts if f["dissipation_total_erg_s"] > 0]
    diss_scale = float(np.quantile(positive_diss, 0.75)) if positive_diss else math.nan
    passage = epochs["pericenter"].get("time_gyr")
    epoch_verified = bool(epochs["pericenter"]["verified"])
    for front in fronts:
        temporal = 0.5 if passage is None else float(np.clip(0.5 + (front["time_gyr"] - passage) / (2 * config.passage_window_gyr), 0, 1))
        spatial = min(1.0, front["max_extent_kpc"] / config.cluster_extent_kpc)
        coherence = front["normal_coherence"]
        if math.isfinite(front["axis_offset_kpc"]):
            orientation = abs(float(_merger_front_normal(front) @ _merger_front_axis(front)))
            axis = math.exp(-0.5 * (front["axis_offset_kpc"] / config.axis_offset_scale_kpc) ** 2) * orientation
        else:
            axis = 0.5  # unavailable geometry supplies neutral evidence
        continuity = min(1.0, counts[front["track_id"]] / config.minimum_track_length)
        birth = births[front["track_id"]]
        front["track_length_outputs"] = counts[front["track_id"]]
        front["track_birth_time_gyr"] = birth["time_gyr"]
        front["track_birth_distance_midpoint_kpc"] = birth["distance_midpoint_kpc"]
        front["track_birth_origin_type"] = birth["propagation_origin_type"]
        front["track_birth_distance_origin_kpc"] = birth["distance_origin_kpc"]
        launch_unknown = birth["propagation_origin_type"] != "two_center_midpoint"
        origin = 0.5 if launch_unknown else math.exp(-0.5 * (birth["distance_midpoint_kpc"] / config.origin_scale_kpc) ** 2)
        if passage is not None and birth["time_gyr"] > passage + config.passage_window_gyr:
            origin = 0.5  # coverage does not show the front's launch region
            launch_unknown = True
        track_motions = motions.get(front["track_id"], [])
        outward_fraction = float(np.mean(np.asarray(track_motions) > 0)) if track_motions else math.nan
        front["outward_step_fraction"] = outward_fraction
        motion_score = 0.5 if len(track_motions) < 2 else outward_fraction
        diss = 0.0 if not positive_diss else float(np.clip(math.log1p(front["dissipation_total_erg_s"]) / math.log1p(diss_scale), 0, 1))
        front.update({"evidence_epoch": temporal, "evidence_extent": spatial,
            "evidence_normal": coherence, "evidence_axis": axis,
            "evidence_continuity": continuity, "evidence_outward_motion": motion_score,
            "evidence_dissipation": diss, "evidence_origin": origin})
        # Weak criteria can be offset by strong independent evidence.
        score = 0.12 * temporal + 0.14 * spatial + 0.14 * coherence + 0.18 * axis + 0.14 * continuity + 0.10 * motion_score + 0.10 * diss + 0.08 * origin
        front["merger_evidence_score"] = float(score)
        flags = front["quality_flags"]
        if not epoch_verified:
            flags.append("pericenter_reference_unverified")
        if not epochs["overlap"]["verified"]:
            flags.append("overlap_reference_unverified")
        if not epochs["apocenter"]["verified"]:
            flags.append("apocenter_reference_unverified")
        if front["outward_motion_basis"] == "unmeasured":
            flags.append("propagation_unmeasured")
        if len(track_motions) < 2:
            flags.append("outward_motion_short_baseline")
        if counts[front["track_id"]] < config.minimum_track_length:
            flags.append("short_track")
        if not positive_diss:
            flags.append("dissipation_reference_unavailable")
        if not config.thresholds_calibrated:
            flags.append("thresholds_not_calibrated")
        if launch_unknown:
            flags.append("launch_region_not_observed")
        if score >= config.candidate_score:
            label = "candidate"
        elif score >= config.uncertain_score:
            label = "uncertain"
        else:
            label = "unclassified"
        front["classification"] = label
        confidence_flags = {"validation_unavailable", "few_cells", "low_normal_coherence",
                            "ambiguous_temporal_link", "short_track", "outward_motion_short_baseline",
                            "launch_region_not_observed", "dissipation_reference_unavailable",
                            "thresholds_not_calibrated", "secondary_center_unavailable",
                            "axis_carried_forward", "merger_axis_unavailable", "propagation_origin_transition"}
        front["uncertain"] = bool(label != "candidate" or not epoch_verified or confidence_flags.intersection(flags))
        front["confidence"] = ("high" if label == "candidate" and not front["uncertain"] and counts[front["track_id"]] >= config.minimum_track_length else
                               "medium" if label == "candidate" else "low")
    return diss_scale if math.isfinite(diss_scale) else None


def _merger_config(info):
    options = copy.deepcopy(dict(info.get("merger_shock_options", {})))
    allowed = {f.name for f in fields(_MergerOptions)} - {
        "center_position_unit", "center_coordinate_frame", "result_coordinate_frame"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unknown merger_shock_options: {sorted(unknown)}")
    return _MergerOptions(center_position_unit=info.get("position_unit", "kpc"),
                  center_coordinate_frame=info.get("coordinate_frame", "physical"),
                  result_coordinate_frame=info.get("result_coordinate_frame", "physical"), **options).validate()


def _merger_same_scientific_options(left, right):
    # Workspace/selection settings can change between outputs without changing
    # front geometry or evidence. Fill defaults for saved version-2 states.
    performance = {"spatial_query_chunk", "cell_chunk_size", "max_neighbor_pairs", "expand_candidate_cells", "neighbor_backend"}
    normalize = lambda values: {k: v for k, v in vars(_MergerOptions(**values)).items() if k not in performance}
    return normalize(left) == normalize(right)


def _merger_column(info, names, n, *, required=True):
    present = [name for name in names if name in info]
    if not present:
        if required:
            raise ValueError(f"cluster_info requires {names[0]!r}")
        return np.full(n, math.nan)
    values = []
    for name in present:
        value = np.asarray(info[name], dtype=float)
        if value.ndim == 0 and n == 1:
            value = value.reshape(1)
        if value.shape != (n,):
            raise ValueError(f"cluster_info[{name!r}] must have shape ({n},)")
        values.append(value)
    if any(not np.array_equal(values[0], value, equal_nan=True) for value in values[1:]):
        raise ValueError(f"conflicting cluster_info aliases {present}")
    return values[0]


def _merger_center_history(iout, info, config):
    first = np.asarray(_merger_required(info, "ccen1"), dtype=float)
    if first.shape == (3,):
        first = first.reshape(1, 3)
    if first.ndim != 2 or first.shape[1] != 3 or not len(first):
        raise ValueError("cluster_info['ccen1'] must have shape (N, 3)")
    n = len(first)
    if n == 1 and "iout" not in info and "snapshot" not in info:
        snapshots = np.array([iout])
    else:
        numbers = _merger_column(info, ("iout", "snapshot"), n)
        if not np.all(np.isfinite(numbers)) or np.any(numbers != np.floor(numbers)):
            raise ValueError("cluster_info snapshot numbers must be finite integers")
        snapshots = numbers.astype(np.int64)
    if len(np.unique(snapshots)) != n or iout not in snapshots:
        raise ValueError("cluster_info snapshot numbers must be unique and include iout")
    if "ccen2" not in info:
        raise ValueError("cluster_info requires 'ccen2'; use NaN after secondary termination")
    second = np.asarray(info["ccen2"], dtype=float) if info["ccen2"] is not None else np.full((n, 3), math.nan)
    if second.shape == (3,) and n == 1:
        second = second.reshape(1, 3)
    if second.shape != first.shape:
        raise ValueError("cluster_info['ccen2'] must match the shape of ccen1")
    available = np.all(np.isfinite(second), axis=1)
    absent = np.all(np.isnan(second), axis=1)
    if not np.all(np.isfinite(first)) or not np.all(available | absent):
        raise ValueError("centers must be finite 3-vectors; an absent secondary needs three NaNs")
    if "secondary_center_available" in info:
        declared = np.asarray(info["secondary_center_available"])
        if declared.ndim == 0 and n == 1:
            declared = declared.reshape(1)
        if declared.shape != (n,) or declared.dtype.kind != "b" or not np.array_equal(declared, available):
            raise ValueError("secondary_center_available must be a boolean array matching ccen2")
    age = _merger_column(info, ("time_gyr", "t_BB"), n)
    z = _merger_column(info, ("redshift",), n)
    r1 = _merger_column(info, ("rvir1", "cluster_rvir"), n, required=False)
    r2 = _merger_column(info, ("rvir2", "cluster_rvir2"), n, required=False)
    if np.any(np.isinf(r1) | np.isinf(r2)) or np.any((np.isfinite(r1) & (r1 <= 0)) | (np.isfinite(r2) & (r2 <= 0))):
        raise ValueError("virial radii must be positive or NaN when unavailable")
    if np.any(absent & np.isfinite(r2)):
        raise ValueError("rvir2 must be NaN after secondary termination")
    order = np.argsort(age)
    if (not np.all(np.isfinite(age)) or np.any(np.diff(age[order]) <= 0) or
            np.any(np.diff(snapshots[order]) <= 0) or not np.all(np.isfinite(z)) or np.any(np.diff(z[order]) > 0)):
        raise ValueError("snapshots and cosmic ages must strictly increase; redshift must not increase")
    ordered_available = available[order]
    if np.any(np.diff(ordered_available.astype(int)) > 0):
        raise ValueError("secondary absence must be terminal; interior gaps cannot establish coalescence")
    if not np.any(available) and np.min(snapshots) <= info.get("secondary_branch_last_snapshot", 875):
        raise ValueError("an entirely primary-only history must start after secondary_branch_last_snapshot (default 875)")
    factor = _MERGER_LENGTH_TO_KPC[config.center_position_unit]
    centers, meta = {}, {}
    for index in order:
        snap = int(snapshots[index])
        radii = tuple(float(r[index]) * factor if np.isfinite(r[index]) else None for r in (r1, r2))
        centers[snap] = (first[index] * factor, second[index] * factor if available[index] else None,
                         radii if any(r is not None for r in radii) else None)
        meta[snap] = {"time_gyr": float(age[index]), "redshift": float(z[index])}
    history = list(centers)
    serial = [{"snapshot": s, **meta[s], "primary": centers[s][0].tolist(),
               "secondary": centers[s][1].tolist() if centers[s][1] is not None else None,
               "radii": centers[s][2]} for s in history]
    signature = hashlib.sha256(json.dumps(serial, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return centers, meta, history, signature


def _merger_reset_front(front):
    front["quality_flags"] = list(front["base_quality_flags"])
    for name in ("front_id", "track_id", "previous_front_id", "next_front_id"):
        front[name] = ""
    for name in ("propagation_x", "propagation_y", "propagation_z", "propagation_speed_kpc_gyr",
                 "propagation_speed_km_s", "outward_axis_speed_kpc_gyr", "outward_radial_speed_kpc_gyr"):
        front[name] = math.nan


def _merger_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _merger_source_fingerprint(rows, input_ids, pos, mach, dx, normal):
    return _merger_fingerprint(np.asarray(rows, dtype="<i8"), np.asarray(input_ids, dtype="<i8"),
                        *(np.asarray(a, dtype="<f8") for a in (pos, mach, dx, normal)))


def _merger_group_fingerprint(rows, data, chunk_size):
    """Same canonical digest as member recovery, without full group copies."""
    digest = hashlib.sha256()
    for name in ("retained_row", "cell_id", "pos", "mach", "dx", "normal"):
        source = data[name]
        dtype = np.dtype("<i8" if name in ("retained_row", "cell_id") else "<f8")
        shape = (len(rows),)+source.shape[1:]
        digest.update(str((dtype.str, shape)).encode())
        for start in range(0, len(rows), chunk_size):
            block = np.ascontiguousarray(source[rows[start:start+chunk_size]], dtype=dtype)
            digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _merger_public_fronts(fronts):
    return [{"shock_id": f["representative_shock_id"], "evidence": f["merger_evidence_score"],
             "evidence_components": {name: f[f"evidence_{name}"] for name in _MERGER_EVIDENCE_NAMES},
             "confidence": f["confidence"], "quality_flags": tuple(dict.fromkeys(f["quality_flags"])),
             "classification": f["classification"], "uncertain": f["uncertain"],
             "track_origin": f["track_id"], "previous_front": f["previous_front_id"],
             "next_front": f["next_front_id"],
             "propagation_speed_km_s": f["propagation_speed_km_s"],
             "outward_axis_speed_kpc_gyr": f["outward_axis_speed_kpc_gyr"],
             "outward_radial_speed_kpc_gyr": f["outward_radial_speed_kpc_gyr"]} for f in fronts]


def _merger_score_sequence(state):
    # Stored base summaries are shared between successive states and never
    # modified. Only this temporary scoring copy grows with the full sequence.
    # Keeping a scored copy of every prefix in every output would use quadratic
    # memory when callers retain their per-snapshot catalogs.
    by_snapshot = {frame["snapshot"]: copy.deepcopy(frame["fronts"]) for frame in state["frames"]}
    entries = [{"snapshot": frame["snapshot"]} for frame in state["frames"]]
    for fronts in by_snapshot.values():
        for front in fronts:
            _merger_reset_front(front)
    config = _MergerOptions(**state["configuration"])
    _merger_link_fronts(by_snapshot, entries, state["metadata"], config)
    scale = _merger_score_fronts(by_snapshot, state["epochs"], config)
    return by_snapshot, scale


def _merger_selection(catalog):
    membership = catalog["membership"]
    assessments = catalog["fronts"]
    candidates = np.array([f["classification"] == "candidate" for f in assessments], dtype=bool)
    expand = catalog["tracking_state"]["configuration"].get("expand_candidate_cells", True)
    if not expand:
        groups = np.flatnonzero(candidates)
        catalog["shock_id"] = np.asarray([assessments[g]["shock_id"] for g in groups], dtype=membership["shock_id"].dtype)
        catalog["selection_mode"] = "front_representatives"
    else:
        labels = membership["front_index"]
        selected = np.zeros(len(labels), dtype=bool)
        valid = labels >= 0
        selected[valid] = candidates[labels[valid]]
        groups = labels[selected]
        catalog["shock_id"] = membership["shock_id"][selected]
        catalog["selection_mode"] = "candidate_cells"
    catalog["evidence"] = np.asarray([a["evidence"] for a in assessments], dtype=float)[groups]
    catalog["confidence"] = np.asarray([a["confidence"] for a in assessments], dtype="U6")[groups]
    # Tuples are shared per front rather than copied per detected cell.
    flag_values = np.empty(len(assessments), dtype=object)
    for i, front in enumerate(assessments):
        flag_values[i] = front["quality_flags"]
    catalog["quality_flags"] = tuple(flag_values[groups])
    catalog["uncertain"] = np.asarray([a["uncertain"] for a in assessments], dtype=bool)[groups]
    return catalog


def make_shock_catalog(
    cell,
    output_dir,
    *,
    snapshot=None,
    region=None,
    minlevel=13,
    maxlevel=20,
    min_mach=1.5,
    gamma=5.0 / 3.0,
    mach_tolerance=0.3,
    normal_cosine=0.7,
    duplicate_normal_cosine=0.8,
    external_temperature=1.0e4,
    classification_fraction=0.8,
    show_progress=True,
    make_qa_plot=True,
):
    """Run the complete catalog workflow for one extracted AMR region.

    Returns
    -------
    analysis:
        ``ShockAnalysis`` containing the cell-level result, dissipation fields,
        catalog, timings, and counts. Call ``analysis.clear()`` after any
        cell-level products are no longer needed.
    paths:
        Paths of the versioned NPZ catalog, summary CSV, and optional QA image.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {}
    if snapshot is not None:
        provenance["snapshot"] = snapshot
    if region is not None:
        provenance["region"] = region

    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    finder.min_mach = min_mach
    finder.gamma = gamma
    finder.boundary = "open"
    finder.show_progress = show_progress

    analysis = finder.analyze(
        cell,
        compute_dissipation=True,
        build_catalog=True,
        deduplicate=True,
        mach_tolerance=mach_tolerance,
        normal_cosine=normal_cosine,
        duplicate_normal_cosine=duplicate_normal_cosine,
        min_mach=min_mach,
        external_temperature=external_temperature,
        classification_fraction=classification_fraction,
        provenance=provenance,
    )

    catalog_path = output_dir / "shock_catalog.npz"
    csv_path = output_dir / "shock_groups.csv"
    qa_path = output_dir / "shock_catalog_qa.png"

    shocktest.save_shock_catalog(catalog_path, analysis.catalog)
    shocktest.save_shock_catalog_csv(csv_path, analysis.catalog)

    summary = shocktest.summarize_catalog_quality(analysis.catalog)
    print("analysis counts:", analysis.counts)
    print("analysis timings [s]:", analysis.timings)
    print("catalog quality:", summary)
    print_group_preview(analysis.catalog)

    paths = {"catalog": catalog_path, "csv": csv_path, "qa": None}
    if make_qa_plot:
        figure, _ = shocktest.plot_catalog_quality(analysis.catalog)
        figure.savefig(qa_path, dpi=180)
        plt.close(figure)
        paths["qa"] = qa_path

    # Demonstrate that the complete catalog can be used without ``cell``.
    loaded = shocktest.load_shock_catalog(catalog_path)
    if loaded.metadata != analysis.catalog.metadata:
        raise RuntimeError("saved catalog metadata failed the round-trip check")

    return analysis, paths


def print_group_preview(catalog, *, limit=10):
    """Print a compact preview of the strongest deterministic group IDs."""

    print(f"shock groups: {len(catalog.groups)}")
    for group in catalog.groups[:limit]:
        flags = ",".join(group.quality_flags) or "none"
        print(
            f"group={group.group_id:5d} "
            f"M_peak={group.mach_peak:7.3f} "
            f"M_mean={group.mach_mean:7.3f} "
            f"area={group.area:.6e} {group.area_unit} "
            f"E_diss={group.dissipation_total:.6e} erg/s "
            f"class={group.classification:10s} "
            f"complete={group.is_complete!s:5s} "
            f"flags={flags}"
        )


def load_catalog_only(path):
    """Load and summarize a previously saved catalog without AMR cell data."""

    catalog = shocktest.load_shock_catalog(path)
    print("metadata:", catalog.metadata)
    print("quality:", shocktest.summarize_catalog_quality(catalog))
    print_group_preview(catalog)
    return catalog


# Typical usage after reading a region from a snapshot:
#
# analysis, paths = make_shock_catalog(
#     cell,
#     "output/shock_catalog_00620",
#     snapshot=620,
#     region="cluster-core",
#     minlevel=13,
#     maxlevel=20,
#     min_mach=1.5,
# )
#
# result = analysis.result
# dissipation = analysis.dissipation
# catalog = analysis.catalog
#
# # Use result/dissipation here for cell-level maps, then release large arrays.
# analysis.clear()
#
# # The compact catalog remains available from disk without the original cell.
# catalog = load_catalog_only(paths["catalog"])


# Merger-only post-processing of existing files (no ShockFinder run):
#
# cluster_info = {
#     "iout": history_snapshot_numbers,
#     "ccen1": primary_centers_physical_kpc,       # shape (N, 3)
#     "ccen2": secondary_centers_physical_kpc,     # NaN after branch termination
#     "rvir1": primary_rvir_physical_kpc,
#     "rvir2": secondary_rvir_physical_kpc,        # NaN after branch termination
#     "t_BB": cosmic_age_gyr,
#     "redshift": snapshot_redshifts,
# }
# catalogs = []
# for iout, result, dissipation in saved_outputs_in_time_order:
#     catalog = merger_shock_catalog(iout, result, dissipation, cluster_info)
#     catalogs.append(catalog)
#     cluster_info["previous_catalog"] = catalog  # explicit temporal context
# catalogs = finalize_merger_shock_catalogs(catalogs)
#
# catalog = catalogs[-1]
# candidate_pos = result.pos[catalog["shock_id"]]
# candidate_mach = result.mach[catalog["shock_id"]]
# if catalog["shock_id"].size:
#     front = get_merger_shock_members(catalog, catalog["shock_id"][0], result, dissipation)
#     input_cell_ids = front["input_cell_id"]
