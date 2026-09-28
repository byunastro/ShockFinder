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
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

import shocktest

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
    """
    if isinstance(iout, bool) or not isinstance(iout, (int, np.integer)):
        raise ValueError("iout must be an integer snapshot number")
    iout = int(iout)
    config = _merger_config(cluster_info)
    centers, meta, history, signature = _merger_center_history(iout, cluster_info, config)
    epochs = _merger_verify_epochs(config, centers, meta, history)
    geoms = _merger_geometry_history(centers, meta, history, config)
    previous = cluster_info.get("previous_catalog")
    if previous is None:
        frames = ()
    else:
        state = previous.get("tracking_state", {})
        if state.get("schema_version") != _MERGER_SCHEMA_VERSION:
            raise ValueError("previous_catalog is not a compatible merger catalog")
        if state["configuration"] != vars(config) or state["center_history_signature"] != signature:
            raise ValueError("center history or options changed; start a new sequence with previous_catalog=None")
        frames = state["frames"]
        if frames and iout <= frames[-1]["snapshot"]:
            raise ValueError("calls with previous_catalog must advance in snapshot order")
        # Reuse immutable history metadata. No retained-cell arrays are shared
        # with the inputs, and no earlier state is updated during scoring.
        meta, epochs = state["metadata"], state["epochs"]
    data = _merger_compact_results(result, dissipation, snapshot=iout)
    n = data["retained_count"]
    center_ids = _merger_array(result, "center_index", n)[data["retained_row"]]
    if center_ids.dtype.kind not in "iu" or not np.array_equal(center_ids, data["retained_row"]):
        raise ValueError("expected dense ShockResult: accepted center_index must equal its retained result row")
    groups = _merger_group_cells(data, config)
    current = []
    labels = np.full(len(data["retained_row"]), -1, dtype=np.int32)
    fingerprints = []
    for group_index, rows in enumerate(groups):
        front = _merger_summarize_front(rows, data, geoms[iout], iout, meta[iout]["time_gyr"], meta[iout]["redshift"], config)
        front["representative_shock_id"] = int(np.min(data["retained_row"][rows]))
        front["base_quality_flags"] = tuple(front["quality_flags"])
        labels[rows] = group_index
        fingerprints.append(_merger_source_fingerprint(data["retained_row"][rows], data["cell_id"][rows],
            data["pos"][rows], data["mach"][rows], data["dx"][rows], data["normal"][rows]))
        del front["rows"]  # No member arrays or dense inputs in tracking state.
        current.append(front)
    frames = frames + ({"snapshot": iout, "fronts": current,
                        "source_fingerprint": tuple(fingerprints)},)
    state = {"schema_version": _MERGER_SCHEMA_VERSION, "configuration": vars(config),
             "center_history_signature": signature, "metadata": meta, "epochs": epochs,
             "frames": frames}
    scored, diss_scale = _merger_score_sequence(state)
    catalog = {"schema_version": _MERGER_SCHEMA_VERSION, "iout": iout,
        "fronts": _merger_public_fronts(scored[iout]),
        "membership": {"shock_id": data["retained_row"], "input_cell_id": data["cell_id"],
                       "front_index": labels, "source_fingerprint": tuple(fingerprints)},
        "source": {"retained_count": n, "position_unit": data["result_position_unit"],
                   "id_space": "dense retained result row / center_index", "snapshot": iout},
        "epoch_verification": epochs, "dissipation_reference_p75_erg_s": diss_scale,
        "tracking_state": state,
    }
    return _merger_selection(catalog)


def get_merger_shock_members(catalog, shock_id, result, dissipation=None):
    """Recover the entire front containing an existing center ID.

    Returned geometry is in the original result's position_unit. Endpoint
    indices remain in dense retained-row space; -1 means unavailable.
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
    scored, diss_scale = _merger_score_sequence(state)
    frames = {frame["snapshot"]: frame for frame in state["frames"]}
    refreshed = []
    for catalog in catalogs:
        own_state = catalog["tracking_state"]
        if (own_state["configuration"] != state["configuration"] or
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
        refreshed.append(_merger_selection(updated))
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
        if not 0 <= config.minimum_neighbor_normal_cosine <= 1:
            raise ValueError("minimum_neighbor_normal_cosine must be in [0, 1]")
        if not 0 <= config.uncertain_score < config.candidate_score <= 1:
            raise ValueError("score cutoffs must satisfy 0 <= uncertain < candidate <= 1")
        if config.box_size_kpc is not None and config.box_size_kpc <= 0:
            raise ValueError("box_size_kpc must be positive")
        if config.post_merger_axis_policy not in {"last_measured", "unavailable"}:
            raise ValueError("post_merger_axis_policy must be 'last_measured' or 'unavailable'")
        return self


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


def _merger_compact_results(result, dissipation, *, snapshot=None):
    """Select saved detections without retaining the dense input object."""
    n = len(_merger_required(result, "mach"))
    unit = _merger_required(result, "position_unit")
    if unit not in _MERGER_LENGTH_TO_KPC:
        raise ValueError(f"unsupported or unspecified result.position_unit {unit!r}")
    factor = _MERGER_LENGTH_TO_KPC[unit]
    shock = _merger_array(result, "shock", n, dtype=bool)
    retained_rows = np.flatnonzero(shock)
    # Compact immediately: real NewCluster pickles can hold hundreds of
    # millions of retained rows. Never convert the full geometry to kpc.
    mach = _merger_array(result, "mach", n)[retained_rows].astype(float, copy=False)
    pos = _merger_array(result, "pos", shape=(n, 3))[retained_rows].astype(float, copy=False) * factor
    dx = _merger_array(result, "dx", n)[retained_rows].astype(float, copy=False) * factor
    normal = _merger_array(result, "normal", shape=(n, 3))[retained_rows].astype(float, copy=False)
    cell_id = _merger_array(result, "selected_indices", n)[retained_rows].astype(np.int64, copy=False)
    valid = np.isfinite(mach) & (mach > 1) & np.all(np.isfinite(pos), axis=1) & np.isfinite(dx) & (dx > 0)
    normal_length = np.linalg.norm(normal, axis=1)
    valid &= np.all(np.isfinite(normal), axis=1) & (normal_length > 0)
    consistent = _merger_value(result, "mach_consistent")
    status = _merger_value(result, "mach_validation_status")
    validation_unknown = consistent is None and status is None
    selected_consistent = None
    selected_status = None
    if consistent is not None:
        selected_consistent = _merger_array(result, "mach_consistent", n, dtype=bool)[retained_rows]
        valid &= selected_consistent
    if status is not None:
        status = np.asarray(status)
        if status.shape != (n,):
            raise ValueError("mach_validation_status shape mismatch")
        selected_status = status[retained_rows].astype(np.int64)
        valid &= (selected_status & (1 << 8)) == 0  # ENDPOINT_INVALID
        if consistent is None:
            valid &= (selected_status & (1 << 7)) != 0  # MACH_CONSISTENT
    endpoint1, endpoint2 = _merger_value(result, "upstream_index"), _merger_value(result, "downstream_index")
    endpoints_valid = None
    if endpoint1 is not None and endpoint2 is not None:
        endpoints_valid = (_merger_array(result, "upstream_index", n)[retained_rows] >= 0) & (_merger_array(result, "downstream_index", n)[retained_rows] >= 0)
        valid &= endpoints_valid
    if len(np.unique(cell_id)) != len(cell_id):
        raise ValueError("shock cell identifiers are not unique within snapshot")
    data = {"retained_count": n, "retained_row": retained_rows, "cell_id": cell_id,
        "shock": np.ones(len(retained_rows), dtype=bool), "valid": valid,
        "mach_consistent": selected_consistent, "mach_validation_status": selected_status,
        "endpoints_valid": endpoints_valid, "mach": mach, "pos": pos, "dx": dx,
        "normal": normal / np.maximum(normal_length[:, None], 1e-300),
        "validation_unknown": validation_unknown, "result_position_unit": unit}
    if dissipation is None:
        raise ValueError("no saved dissipation supplied; do not recompute it")
    flux = _merger_array(dissipation, "flux", n)[retained_rows].astype(float, copy=False)
    total = _merger_array(dissipation, "total", n)[retained_rows].astype(float, copy=False)
    area = _merger_array(dissipation, "area", n)[retained_rows].astype(float, copy=False)
    data["valid"] &= np.isfinite(flux) & (flux >= 0) & np.isfinite(total) & (total >= 0) & np.isfinite(area) & (area > 0)
    data.update({"flux": flux, "total": total, "area": area, "snapshot": snapshot})
    return data


class _MergerUnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = int(self.parent[i])
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _merger_group_cells(data, config):
    """Connect AMR cell footprints with nearby, similarly oriented normals."""
    rows = np.flatnonzero(data["valid"])
    if len(rows) == 0:
        return []
    pos, dx, normals = data["pos"][rows], data["dx"][rows], data["normal"][rows]
    labels = _MergerUnionFind(len(rows))
    # AMR-scale trees keep fine-cell searches local even when a few coarse
    # cells are present in the same snapshot.
    levels = np.floor(np.log2(dx / np.min(dx)) + 1e-8).astype(int)
    buckets = []
    for level in np.unique(levels):
        indices = np.flatnonzero(levels == level)
        buckets.append((indices, cKDTree(pos[indices]), float(np.max(dx[indices]))))
    for a, (source_indices, _, _) in enumerate(buckets):
        for b in range(a, len(buckets)):
            target_indices, tree, target_max_dx = buckets[b]
            for start in range(0, len(source_indices), config.spatial_query_chunk):
                chunk = source_indices[start:start + config.spatial_query_chunk]
                # Vectorized KD queries avoid millions of Python/C crossings.
                radii = math.sqrt(3) * (0.5 * (dx[chunk] + target_max_dx) +
                    config.max_cell_gap_factor * np.maximum(dx[chunk], target_max_dx))
                neighborhoods = tree.query_ball_point(pos[chunk], radii)
                for i, local_hits in zip(chunk, neighborhoods):
                    if not local_hits:
                        continue
                    js = target_indices[np.asarray(local_hits, dtype=np.int64)]
                    if a == b:
                        js = js[js > i]
                    if not len(js):
                        continue
                    reach = 0.5 * (dx[i] + dx[js]) + config.max_cell_gap_factor * np.maximum(dx[i], dx[js])
                    close = np.all(np.abs(pos[js] - pos[i]) <= reach[:, None], axis=1)
                    aligned = np.abs(normals[js] @ normals[i]) >= config.minimum_neighbor_normal_cosine
                    for j in js[close & aligned]:
                        labels.union(int(i), int(j))
    roots = np.fromiter((labels.find(i) for i in range(len(rows))), dtype=np.int64, count=len(rows))
    order = np.argsort(roots, kind="stable")
    cuts = np.flatnonzero(np.diff(roots[order])) + 1
    return [rows[group] for group in np.split(order, cuts)]


def _merger_weighted_normal(normals, weights):
    # Axis signs can differ locally; use an axial mean for orientation only.
    reference = normals[np.argmax(weights)]
    aligned = normals * np.where(normals @ reference < 0, -1.0, 1.0)[:, None]
    mean = np.average(aligned, axis=0, weights=weights)
    coherence = float(np.linalg.norm(mean))
    return mean / max(coherence, 1e-300), coherence


def _merger_summarize_front(rows, data, geom, snapshot, time_gyr, redshift, config):
    pos, dx, area = data["pos"][rows], data["dx"][rows], data["area"][rows]
    weights = area / area.sum()
    center = np.sum(pos * weights[:, None], axis=0)
    lower = np.min(pos - dx[:, None] / 2, axis=0)
    upper = np.max(pos + dx[:, None] / 2, axis=0)
    extent = upper - lower
    normal, coherence = _merger_weighted_normal(data["normal"][rows], area)
    origin_vector = _merger_minimum_image(center - geom["origin"], config.box_size_kpc)
    axis_available = np.all(np.isfinite(geom["axis"]))
    axis_coordinate = float(origin_vector @ geom["axis"]) if axis_available else math.nan
    axis_offset = float(np.linalg.norm(origin_vector - axis_coordinate * geom["axis"])) if axis_available else math.nan
    distances = [float(np.linalg.norm(_merger_minimum_image(center - geom[f"center{i}"], config.box_size_kpc)))
                 if geom[f"center{i}"] is not None else math.nan for i in (1, 2)]
    second_center = geom["center2"] if geom["center2"] is not None else np.full(3, math.nan)
    mach = data["mach"][rows]
    total = data["total"][rows]
    flux = data["flux"][rows]
    flags = list(geom["quality_flags"])
    if len(rows) < config.minimum_front_cells:
        flags.append("few_cells")
    if data["validation_unknown"]:
        flags.append("validation_unavailable")
    if coherence < config.minimum_neighbor_normal_cosine:
        flags.append("low_normal_coherence")
    if np.any(np.sign(data["normal"][rows] @ normal) != np.sign(data["normal"][rows[0]] @ normal)):
        flags.append("normal_signs_mixed")
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
        "mach_min": float(np.min(mach)), "mach_p10": float(np.quantile(mach, 0.1)),
        "mach_median": float(np.median(mach)), "mach_p90": float(np.quantile(mach, 0.9)),
        "mach_max": float(np.max(mach)),
        "dissipation_total_erg_s": float(np.sum(total)),
        "dissipation_median_erg_s": float(np.median(total)),
        "dissipation_flux_median_erg_s_kpc2": float(np.median(flux)),
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
    labels = membership["front_index"]
    selected = np.zeros(len(labels), dtype=bool)
    valid = labels >= 0
    selected[valid] = candidates[labels[valid]]
    groups = labels[selected]
    catalog["shock_id"] = membership["shock_id"][selected]
    catalog["evidence"] = np.asarray([a["evidence"] for a in assessments], dtype=float)[groups]
    catalog["confidence"] = np.asarray([a["confidence"] for a in assessments], dtype="U6")[groups]
    # Tuples are shared per front rather than copied per detected cell.
    catalog["quality_flags"] = tuple(assessments[g]["quality_flags"] for g in groups)
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
