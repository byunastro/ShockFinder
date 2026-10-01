"""Galaxy/shock geometry and time-dependent, rule-based stripping analysis.

``galaxy_stripping_catalog`` consumes saved histories and merger catalogs;
it never runs ShockFinder. The older snapshot geometry examples below remain
available separately.

This example assumes:
- ``cell`` is the AMR gas cell table used by ``shocktest.ShockFinder``.
- ``galaxy_pos_prev`` and ``galaxy_pos_now`` are ``(ngal, 3)`` arrays in km.
- The same galaxy order is used at the previous and current snapshots.

The classification is geometric: a galaxy is marked as crossed when its segment
between two snapshots changes sign across the nearest shock plane and passes
close enough to that shock cell.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import shocktest
from shocktest import pyShockFinder
from shocktest.spatial import nearest_points, connected_components, nearest_segment_shock


KPC_IN_KM = 3.0856775814913673e16
_GYR_IN_S = 1.e9 * 365.25 * 86400.
_LENGTH_TO_KPC = {"km": 1. / KPC_IN_KM, "kpc": 1., "Mpc": 1000.}
# Common column names for explicit selections; these are not mandatory inputs.
GAS_TRACERS = (
    "m_ism", "m_gas_r90", "mcold_gas_r50", "mcold_gas_r90",
    "HI_gas_mass_r50", "HI_gas_mass_r90",
)
STRIPPING_CATEGORIES = (
    "merger_shock_stripping_candidate", "ordinary_infall_rps_candidate",
    "mixed_or_ambiguous", "no_strong_stripping",
)


@dataclass(frozen=True)
class StrippingOptions:
    """Starting thresholds, NOT calibrated probabilities or universal cuts.

    ``tracers`` selects exact history column names; a string or a nonempty
    list/tuple is accepted. Every selected tracer is analyzed independently.
    One significant tracer episode can support stripping; neither tracer
    counts/families nor P_ram enter detection, scores, or confidence.

    Times/widths are Gyr, rates are inverse Gyr, distances are physical kpc.
    Smoothing is a centered log-mass median within a TOTAL width, restricted
    to contiguous resolved samples; zero disables it. A rapid episode needs
    both the fractional-loss and peak-rate thresholds within max duration.
    Its boundaries follow rates > onset_rate_fraction * min_loss_rate_gyr.
    Gaps greater than either max_gap_gyr or gap_factor * median cadence break
    all derivatives, smoothing, pericenters, and interpolated crossings.

    Association half-windows use max(window_gyr, association_cadences * local
    cadence). Cadence exceeding max_association_window_gyr is unresolved.
    Evidence scores use fixed weights in _stripping_assess_episode; score_cut and
    score_margin control decisions. Uncertain encounters block an ordinary
    RPS attribution. The sensitivity sweep varies one option at a time.
    agreement_window_gyr groups coincident events for diagnostics only; their
    timing agreement never changes a score. All significant episodes enter
    the galaxy classification. Conflicting mechanisms or uncertain episodes
    remain mixed. A no-strong-stripping assessment requires adequate sampling
    and min_mass_history_coverage for each selected tracer.
    Pericenter confidence measures the
    smaller inbound/outbound distance rise within window_gyr, relative to
    pericenter_depth_fraction * minimum distance (with a 1 kpc noise scale).
    Turning points below pericenter_confidence_cut cannot support RPS.

    Cell patches are circumscribed disks of radius sqrt(3)/2 * dx. Their
    normal half-width is the projected cube half-width plus
    zone_width_factor * zone_width * dx. Gas apertures dilate these patches.
    ``gas_radius`` is preferred; r90 is explicitly flagged as an aperture
    proxy if no gas_radius is provided. No AMR length is inferred from a level
    without the saved dx. Cell correspondence uses measured front translation
    and a residual <= patch_match_cells * mean dx, never equality of row IDs.
    """

    tracers: str | tuple[str, ...] | list[str] = ("m_gas_r90",)
    smoothing_width_gyr: float = .03
    min_fractional_loss: float = .30
    min_loss_rate_gyr: float = 2.
    onset_rate_fraction: float = .25
    max_episode_duration_gyr: float = .60
    rebound_fraction: float = .50
    min_valid_samples: int = 4
    max_gap_gyr: float = .25
    gap_factor: float = 3.
    window_gyr: float = .15
    association_cadences: float = 1.5
    max_association_window_gyr: float = .40
    agreement_window_gyr: float = .10
    score_cut: float = .65
    score_margin: float = .12
    encounter_score_cut: float = .65
    min_front_evidence: float = .35
    normal_cosine_min: float = .80
    patch_match_cells: float = 2.
    zone_width_factor: float = .50
    max_motion_km_s: float = 1.e4
    min_shock_coverage: float = .80
    min_mass_history_coverage: float = .80
    pericenter_confidence_cut: float = .40
    pericenter_depth_fraction: float = .05
    sensitivity_fractions: tuple = (.20, .30, .40)
    sensitivity_rates_gyr: tuple = (1., 2., 4.)
    sensitivity_smoothing_gyr: tuple = (0., .03, .06)
    sensitivity_windows_gyr: tuple = (.10, .15, .25)
    sensitivity_min_stability: float = .60


def galaxy_stripping_catalog(galaxy_histories, merger_shocks, cluster_info=None,
                            *, options=None, output_dir=None, make_plots=True):
    """Measure gas loss and assess plausible stripping mechanisms.

    Parameters
    ----------
    galaxy_histories : mapping
        Persistent galaxy/branch ID -> mapping of snapshot-aligned arrays.
        Do NOT use an evolving hmid as a persistent key. Supply ``iout``,
        ``t_BB`` (cosmic Gyr), ``gal_cen`` (N,3), and gas fields when available.
        All other fields, including hmid/first/last, masses, SFRs, angles,
        ism_flag/icm_flag and tail fields, are preserved without mutation.

        Each history MUST declare ``units`` with ``time='Gyr'``,
        ``position`` and ``radius`` in km/kpc/Mpc, and
        ``coordinate_frame='physical'``. vx/vy/vz require
        ``velocity='km/s'`` (common Cartesian simulation frame). Their angle
        is reported as a stored-velocity angle; crossing geometry uses the
        measured galaxy motion relative to the moving front instead.
        Gas masses may use any consistent unit PER tracer. Each column named
        in options.tracers must exist and have shape (N,). The default is only
        m_gas_r90. Other gas fields are preserved without being analyzed.
        P_ram is optional and used only in diagnostic plots, never in episode
        detection, mechanism scores, confidence, or sensitivity decisions.

        Optional ``resolved`` and ``valid`` map tracer -> boolean (N,) masks.
        Optional ``mass_limits`` maps tracer -> nonnegative scalar/(N,) upper
        limits in that tracer's mass unit. Zeros or positive masses <= a
        supplied limit are censored. Only supplied POSITIVE upper limits may
        enter a logarithmic lower bound on loss rate. Unknown zeros, negative
        values, missing/unresolved samples never enter a logarithm. Unknown
        resolution is flagged. Raw invalid/censored values are retained.
        Supply ``gas_radius`` or (flagged proxy) ``r90``. ``host_cluster`` may
        be 1/2 or (N,); otherwise infer an initial host from distance/rvir,
        keeping that host until its branch terminates. Host changes break
        orbital turning-point detection.
    merger_shocks : mapping or callable
        Snapshot -> product, or callable(iout) -> product/None for streaming.
        A product is ``{'members': [independently_assessed_front_members], ...}``.
        Generic shock_front_catalog arrays contain no merger attribution and
        are not accepted as independently assessed merger shocks.
        Front assessments MUST carry independently established classification,
        evidence and a snapshot-local front_id, or an independently supplied
        track_origin for crossing measurements across outputs. Snapshot-local
        IDs support proximity only; they do not establish temporal crossings.
        No gas history is used to change merger-shock attribution.

        Set ``complete=True`` ONLY if the saved search covers the galaxies'
        positions/trajectories in this output. Optional ``covered_galaxy_ids``
        restricts that assertion. Missing products/unspecified coverage are
        unknown exposure, never non-exposure. Empty covered products are valid.
        Supplied members are compacted one output at a time. Products are not
        retained in the return value. Cell/patch IDs are not assumed persistent.
        Optional ``provenance`` records source paths/checksums verbatim in the
        output; rerunning geometry requires the saved cell data.
    cluster_info : mapping, optional
        Independently supplied cluster history: iout, t_BB, ccen1/ccen2,
        optional rvir1/rvir2 and redshift. position_unit defaults to physical
        kpc following the supplied NewCluster conversion. NaN secondary
        centers after branch termination are allowed. Optional box_size is a
        scalar, (3,), or (N,3) physical box length in position_unit; periodic
        boundaries are used in all distances and relative trajectories.
        A history-specific cluster_info can be provided instead.
    options : StrippingOptions or dict, optional
        Overrides documented defaults. Unknown options fail explicitly.
    output_dir : path, optional
        Save the complete analysis pickle, CSV catalogs, JSON configuration,
        and optional PNG diagnostics. No README or original data are edited.

    Returns
    -------
    dict
        gas_loss_events, classifications, shock_encounters, pericenters,
        episode_assessments, sensitivity, population, diagnostic_series,
        configuration and input_histories. Tables are lists of row dictionaries
        (pandas.DataFrame(...) is optional). Every selected tracer has its own
        status, events, and episode assessments in gas_loss_by_tracer.
        Galaxy decisions consider ALL significant selected-tracer episodes,
        without voting or requiring multiple tracers to lose gas. Different
        mechanisms or an unresolved episode produce mixed_or_ambiguous.
        The top-level loss/timing measurements describe the largest-loss
        episode, identified by summary_gas_event_id; all episodes are retained.
        Evidence scores are indices in [0,1], NOT membership probabilities.
        ``causal_confirmation`` is always False. Gas loss alone cannot rule
        out phase changes, consumption, tides, or aperture changes.

    Example
    -------
    >>> analysis = galaxy_stripping_catalog(
    ...     {branch_id: history}, products_by_iout, cluster_info,
    ...     options={"tracers": ["m_gas_r90", "HI_gas_mass_r90"],
    ...              "min_fractional_loss": .3}, output_dir="stripping_output")
    >>> events = analysis["gas_loss_events"]
    >>> category = analysis["classifications"][0]["category"]
    """
    cfg = _stripping_options(options)
    if not hasattr(galaxy_histories, "items"):
        raise TypeError("galaxy_histories must map persistent galaxy IDs to histories")
    series = {}
    shared_clusters = _stripping_clusters(cluster_info) if cluster_info is not None else None
    for galaxy_id, history in galaxy_histories.items():
        series[galaxy_id] = _stripping_history(
            galaxy_id, history, cluster_info, cfg, shared_clusters=shared_clusters)
    encounters, provenance = _stripping_associate_shocks(series, merger_shocks, cfg)
    result = {"schema_version": 2, "configuration": asdict(cfg),
              "input_histories": galaxy_histories, "diagnostic_series": series,
              "shock_encounters": encounters, "shock_provenance": provenance, "gas_loss_events": [],
              "classifications": [], "episode_assessments": [],
              "pericenters": [], "sensitivity": [], "causal_confirmation": False,
              "interpretation": "temporal association and modeled exposure; evidence for enhanced stripping, not proof of causality"}
    for gid, data in series.items():
        row, events, assessments = _stripping_analyze_galaxy(gid, data, cfg)
        variants = _stripping_sensitivity(gid, data, cfg, row)
        stability = np.mean([v["category"] == row["category"] for v in variants]) if variants else 1.
        row["classification_stability"] = float(stability)
        row["baseline_category"] = row["category"]
        if stability < cfg.sensitivity_min_stability:
            row["ambiguity_reasons"].append("classification_sensitive_to_thresholds")
            row["category"] = "mixed_or_ambiguous"
            row["confidence"] = "low"
        elif stability < 1. and row["confidence"] == "high":
            row["confidence"] = "medium"
        row["quality_flags"] = sorted(set(row["quality_flags"] + ["uncalibrated_evidence_scores"]))
        result["classifications"].append(row)
        result["gas_loss_events"].extend(events)
        result["episode_assessments"].extend(assessments)
        result["pericenters"].extend(data["pericenters"])
        result["sensitivity"].extend(variants)
    result["population"] = _stripping_population(result)
    if output_dir is not None:
        result["paths"] = save_galaxy_stripping_analysis(result, output_dir, make_plots=make_plots)
    return result


def _stripping_options(options):
    cfg = options if isinstance(options, StrippingOptions) else StrippingOptions(**(options or {}))
    tracers = (cfg.tracers,) if isinstance(cfg.tracers, str) else cfg.tracers
    if (not isinstance(tracers, (list, tuple)) or not tracers
            or any(not isinstance(name, str) or not name.strip() for name in tracers)):
        raise ValueError("tracers must be a column name or a nonempty list/tuple of column names")
    if len(set(tracers)) != len(tracers):
        raise ValueError("tracers must contain unique column names")
    cfg = replace(cfg, tracers=tuple(tracers))
    bounded = ("min_fractional_loss", "onset_rate_fraction", "rebound_fraction", "score_cut",
               "score_margin", "encounter_score_cut", "min_front_evidence", "normal_cosine_min",
               "min_shock_coverage", "min_mass_history_coverage", "pericenter_confidence_cut",
               "pericenter_depth_fraction", "sensitivity_min_stability")
    for key in bounded:
        if not np.isfinite(getattr(cfg, key)) or not 0 < getattr(cfg, key) <= 1:
            raise ValueError(f"{key} must be in (0,1]")
    for key in ("min_loss_rate_gyr", "max_episode_duration_gyr", "max_gap_gyr", "gap_factor",
                "window_gyr", "association_cadences", "max_association_window_gyr",
                "agreement_window_gyr", "patch_match_cells", "max_motion_km_s"):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("smoothing_width_gyr", "zone_width_factor"):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if (isinstance(cfg.min_valid_samples, bool) or not isinstance(cfg.min_valid_samples, (int, np.integer))
            or cfg.min_valid_samples < 1):
        raise ValueError("invalid min_valid_samples")
    for field, grid in (("min_fractional_loss", cfg.sensitivity_fractions), ("min_loss_rate_gyr", cfg.sensitivity_rates_gyr),
                        ("smoothing_width_gyr", cfg.sensitivity_smoothing_gyr), ("window_gyr", cfg.sensitivity_windows_gyr)):
        for value in grid:
            if not np.isfinite(value) or value < 0 or (field != "smoothing_width_gyr" and value == 0) or (field == "min_fractional_loss" and value > 1):
                raise ValueError(f"invalid sensitivity grid for {field}")
    return cfg


def _stripping_column(mapping, name, n, *, default=np.nan, shape=None):
    arr = np.asarray(mapping.get(name, np.full(shape or (n,), default)), dtype=float)
    if arr.shape != (shape or (n,)):
        raise ValueError(f"{name} must have shape {shape or (n,)}")
    return arr.copy()


def _stripping_delta(delta, box):
    return delta if box is None else delta - box * np.floor(delta / box + .5)


def _stripping_cadence(t, cfg):
    dt = np.diff(t)
    median = float(np.median(dt)) if len(dt) else np.nan
    limit = min(cfg.max_gap_gyr, cfg.gap_factor * median) if len(dt) else cfg.max_gap_gyr
    return median, limit


def _stripping_clusters(info):
    if info is None:
        raise ValueError("cluster_info is required globally or in each galaxy history")
    snapshots = np.asarray(info.get("iout", info.get("snapshot")))
    if snapshots.ndim != 1 or snapshots.dtype.kind not in "iu" or len(np.unique(snapshots)) != len(snapshots):
        raise ValueError("cluster_info.iout must be unique integer snapshot numbers")
    n = len(snapshots)
    factor = _LENGTH_TO_KPC.get(info.get("position_unit", "kpc"))
    if factor is None or info.get("coordinate_frame", "physical") != "physical":
        raise ValueError("cluster coordinates require physical km/kpc/Mpc")
    times = _stripping_column({"t_BB": info.get("t_BB", info.get("time_gyr"))}, "t_BB", n)
    if not np.all(np.isfinite(times)) or len(np.unique(times)) != n:
        raise ValueError("cluster cosmic times must be finite and unique")
    box = info.get("box_size")
    if box is not None:
        box = np.asarray(box, dtype=float) * factor
        if box.ndim == 0:
            box = np.full((n, 3), box)
        elif box.shape == (3,):
            box = np.tile(box, (n, 1))
        if box.shape != (n, 3) or not np.all(np.isfinite(box)) or np.any(box <= 0):
            raise ValueError("box_size must be positive scalar, (3,), or (N,3)")
    centers1 = _stripping_column(info, "ccen1", n, shape=(n, 3)) * factor
    centers2 = _stripping_column(info, "ccen2", n, shape=(n, 3)) * factor
    radii1 = _stripping_column({"r": info.get("rvir1", info.get("cluster_rvir", np.full(n, np.nan)))}, "r", n) * factor
    radii2 = _stripping_column({"r": info.get("rvir2", info.get("cluster_rvir2", np.full(n, np.nan)))}, "r", n) * factor
    order = np.argsort(times)
    sep = np.array([np.linalg.norm(_stripping_delta(centers2[j]-centers1[j], None if box is None else box[j])) for j in order])
    paired = np.isfinite(sep)
    # No interpolated secondary center, and no branch termination at zero separation.
    turns = [j for j in range(1, n-1) if paired[j-1:j+2].all() and sep[j] < sep[j-1] and sep[j] <= sep[j+1]]
    epoch = {"core_passage_time_gyr": float(times[order[turns[0]]]) if turns else np.nan,
             "verified": bool(turns), "source": "bracketed two-center separation minimum" if turns else "unresolved two-center turning point"}
    return {"snapshots": snapshots, "times": times, "c1": centers1, "c2": centers2,
            "r1": radii1, "r2": radii2, "box": box, "epoch": epoch}


def _stripping_history(gid, history, cluster_info, cfg, *, shared_clusters=None):
    missing = [name for name in cfg.tracers if name not in history]
    if missing:
        raise ValueError(f"galaxy {gid}: selected tracer columns missing from history: {missing}")
    units = history.get("units", {})
    if (units.get("time") != "Gyr" or units.get("coordinate_frame") != "physical"
            or units.get("position") not in _LENGTH_TO_KPC or units.get("radius") not in _LENGTH_TO_KPC):
        raise ValueError(f"galaxy {gid}: declare physical position/radius units and time='Gyr' in units")
    snapshots = np.asarray(history["iout"])
    if snapshots.ndim != 1 or snapshots.dtype.kind not in "iu" or len(np.unique(snapshots)) != len(snapshots):
        raise ValueError(f"galaxy {gid}: iout must contain unique integer snapshots")
    n = len(snapshots)
    for tracer in cfg.tracers:
        if np.shape(history[tracer]) != (n,):
            raise ValueError(f"galaxy {gid}: {tracer} must have shape {(n,)}")
    times = _stripping_column(history, "t_BB", n)
    if not np.all(np.isfinite(times)) or len(np.unique(times)) != n or n == 0:
        raise ValueError(f"galaxy {gid}: cosmic times must be finite, unique and nonempty")
    order = np.argsort(times)
    snapshots, times = snapshots[order].copy(), times[order]
    if np.any(np.diff(snapshots) <= 0):
        raise ValueError("snapshot numbers must advance with cosmic time")
    pos = _stripping_column(history, "gal_cen", n, shape=(n, 3))[order] * _LENGTH_TO_KPC[units["position"]]
    clusters = (_stripping_clusters(history["cluster_info"]) if "cluster_info" in history
                else shared_clusters if shared_clusters is not None else _stripping_clusters(cluster_info))
    lookup = {int(s): j for j, s in enumerate(clusters["snapshots"])}
    try:
        ci = np.array([lookup[int(s)] for s in snapshots])
    except KeyError as exc:
        raise ValueError(f"missing cluster metadata for snapshot {exc.args[0]}") from exc
    if not np.allclose(times, clusters["times"][ci], atol=1.e-6, rtol=0.):
        raise ValueError("galaxy and cluster times disagree for the same snapshot")
    box = None if clusters["box"] is None else clusters["box"][ci]
    distances = []
    for center in (clusters["c1"][ci], clusters["c2"][ci]):
        distances.append(np.linalg.norm(_stripping_delta(pos-center, box), axis=1))
    flags = []
    host = history.get("host_cluster")
    if host is None:
        both_radii = (np.isfinite(clusters["r1"][ci]) & (clusters["r1"][ci] > 0)
                      & np.isfinite(clusters["r2"][ci]) & (clusters["r2"][ci] > 0))
        metric = [distances[k] / np.where(both_radii, clusters[f"r{k+1}"][ci], 1.) for k in (0, 1)]
        initial = next((j for j in range(n) if np.isfinite(metric[0][j]) or np.isfinite(metric[1][j])), None)
        h = 1 if initial is None or np.nan_to_num(metric[0][initial], nan=np.inf) <= np.nan_to_num(metric[1][initial], nan=np.inf) else 2
        host = np.full(n, h)
        flags.append("host_inferred_from_initial_distance")
        if h == 2:
            missing = ~np.isfinite(distances[1])
            if np.any(missing):
                host[np.flatnonzero(missing)[0]:] = 1
                flags.append("secondary_branch_ended_host_changed")
    else:
        host = np.asarray(host)
        host = np.full(n, host) if host.ndim == 0 else host[order].copy()
        if host.shape != (n,) or not np.all(np.isin(host, (1, 2))):
            raise ValueError("host_cluster must be 1/2 or a snapshot-aligned array of 1/2")
    selected_distance = np.where(host == 1, distances[0], distances[1])
    radius_field = "gas_radius" if "gas_radius" in history else "r90"
    radius = _stripping_column(history, radius_field, n)[order] * _LENGTH_TO_KPC[units["radius"]]
    if radius_field != "gas_radius":
        flags.append("r90_as_gas_aperture_proxy")
    radius_valid = np.isfinite(radius) & (radius > 0)
    if not radius_valid.all():
        flags.append("gas_radius_missing_or_invalid")
    velocity = np.full((n, 3), np.nan)
    if any(k in history for k in ("vx", "vy", "vz")):
        if units.get("velocity") != "km/s":
            raise ValueError("vx/vy/vz require explicitly declared velocity='km/s'")
        velocity = np.column_stack([_stripping_column(history, k, n)[order] for k in ("vx", "vy", "vz")])
    cadence, gap_limit = _stripping_cadence(times, cfg)
    if np.any(np.diff(times) > gap_limit):
        flags.append("history_time_gaps")
    if not np.all(np.isfinite(pos)):
        flags.append("galaxy_position_gaps")
    if not np.all(np.isfinite(selected_distance)):
        flags.append("cluster_center_gaps")
    data = {"galaxy_id": gid, "iout": snapshots, "time_gyr": times, "order": order,
            "pos_kpc": pos, "radius_kpc": radius, "radius_valid": radius_valid,
            "velocity_km_s": velocity, "box_kpc": box, "host_cluster": host,
            "cluster_distance_kpc": selected_distance, "primary_distance_kpc": distances[0],
            "secondary_distance_kpc": distances[1], "merger_epoch": clusters["epoch"],
            "raw": history,
            "cadence_gyr": cadence, "gap_limit_gyr": gap_limit, "quality_flags": flags}
    data["pericenters"] = _stripping_pericenters(data, cfg)
    return data


def _stripping_pericenters(data, cfg):
    t, distance, host = data["time_gyr"], data["cluster_distance_kpc"], data["host_cluster"]
    rows = []
    for j in range(1, len(t)-1):
        if (host[j-1] != host[j] or host[j] != host[j+1]
                or not np.all(np.isfinite(distance[j-1:j+2]))
                or np.any(np.diff(t[j-1:j+2]) > data["gap_limit_gyr"])):
            continue
        if distance[j] < distance[j-1] and distance[j] <= distance[j+1]:
            left = np.flatnonzero((t < t[j]) & (t >= t[j]-cfg.window_gyr) & (host == host[j]) & np.isfinite(distance))
            right = np.flatnonzero((t > t[j]) & (t <= t[j]+cfg.window_gyr) & (host == host[j]) & np.isfinite(distance))
            depth = min(np.max(distance[left])-distance[j], np.max(distance[right])-distance[j]) if len(left) and len(right) else 0.
            # Very shallow discrete turning points are retained with low evidence.
            confidence = float(np.clip(depth / max(cfg.pericenter_depth_fraction*distance[j], 1.), 0., 1.))
            rows.append({"galaxy_id": data["galaxy_id"], "snapshot": int(data["iout"][j]),
                         "time_gyr": float(t[j]), "bracket_start_gyr": float(t[j-1]),
                         "bracket_end_gyr": float(t[j+1]), "distance_kpc": float(distance[j]),
                         "host_cluster": int(host[j]), "confidence_score": confidence,
                         "verified_turning_point": True,
                         "time_uncertainty_gyr": .5*max(t[j]-t[j-1], t[j+1]-t[j]),
                         "quality_flags": ["sampled_orbital_turning_point"]})
    if not rows:
        finite = np.flatnonzero(np.isfinite(distance))
        if len(finite):
            j = finite[np.argmin(distance[finite])]
            rows.append({"galaxy_id": data["galaxy_id"], "snapshot": int(data["iout"][j]),
                         "time_gyr": float(t[j]), "distance_kpc": float(distance[j]),
                         "host_cluster": int(host[j]), "confidence_score": 0.,
                         "verified_turning_point": False, "time_uncertainty_gyr": np.nan,
                         "quality_flags": ["orbital_minimum_not_bracketed"]})
    return rows


def _stripping_mask(history, key, tracer, n, order):
    mask = history.get(key, {}).get(tracer)
    if mask is None:
        return np.ones(n, bool), False
    mask = np.asarray(mask)
    if mask.shape != (n,) or mask.dtype.kind != "b":
        raise ValueError(f"{key}[{tracer!r}] must be a boolean (N,) array")
    return mask[order].copy(), True


def _stripping_mass_series(data, tracer, cfg):
    n, history, order = len(data["time_gyr"]), data["raw"], data["order"]
    mass = _stripping_column(history, tracer, n)[order]
    valid, _ = _stripping_mask(history, "valid", tracer, n, order)
    resolved, known_resolution = _stripping_mask(history, "resolved", tracer, n, order)
    limits = np.asarray(history.get("mass_limits", {}).get(tracer, np.zeros(n)), dtype=float)
    if limits.ndim == 0:
        limits = np.full(n, limits)
    if limits.shape != (n,) or not np.all(np.isfinite(limits)) or np.any(limits < 0):
        raise ValueError(f"mass_limits[{tracer!r}] must be finite nonnegative scalar or (N,)")
    limits = limits[order]
    flags = []
    status = np.full(n, "missing", dtype="U28")
    finite = np.isfinite(mass)
    status[finite & (mass < 0)] = "invalid_negative"
    status[finite & (mass == 0)] = "zero_unknown_limit"
    status[finite & (mass > 0)] = "unresolved"
    status[finite & ~valid] = "invalid_mask"
    # Only explicit limits support censoring; negative/invalid-mask entries do not.
    censored = finite & valid & (mass >= 0) & (limits > 0) & ((mass <= limits) | ~resolved)
    positive = finite & valid & resolved & (mass > 0) & (mass > limits)
    status[censored] = "upper_limit"
    status[positive] = "resolved_positive"
    if not known_resolution:
        flags.append("resolution_not_supplied")
    if np.any(status == "invalid_negative"):
        flags.append("negative_masses")
    if np.any(status == "zero_unknown_limit"):
        flags.append("zeros_without_positive_upper_limits")
    if np.any(censored):
        flags.append("censored_mass_samples")
    if np.count_nonzero(positive) < cfg.min_valid_samples:
        flags.append("insufficient_resolved_mass_samples")
    t = data["time_gyr"]
    smooth = mass.copy()
    smooth[~positive] = np.nan
    log_mass = np.full(n, np.nan)
    log_mass[positive] = np.log(mass[positive])
    # Restrict windows to contiguous valid segments; never bridge an invalid bin.
    segments = []
    start = None
    for j in range(n):
        if not positive[j] or (start is not None and j > 0 and t[j]-t[j-1] > data["gap_limit_gyr"]):
            if start is not None:
                segments.append((start, j))
            start = None
        if positive[j] and start is None:
            start = j
    if start is not None:
        segments.append((start, n))
    for start, end in segments:
        for j in range(start, end):
            within = (t[start:end] >= t[j]-.5*cfg.smoothing_width_gyr) & (t[start:end] <= t[j]+.5*cfg.smoothing_width_gyr)
            smooth[j] = np.exp(np.median(log_mass[start:end][within]))
    rate = np.full(max(n-1, 0), np.nan)
    rate_bound = np.zeros(max(n-1, 0), bool)
    for j, dt in enumerate(np.diff(t)):
        if dt > data["gap_limit_gyr"] or not positive[j]:
            continue
        if positive[j+1]:
            rate[j] = (np.log(smooth[j])-np.log(smooth[j+1])) / dt
        elif censored[j+1]:
            # Log only the supplied positive upper limit, not the invalid mass.
            rate[j] = (np.log(smooth[j])-np.log(limits[j+1])) / dt
            rate_bound[j] = True
    if np.any(~positive & ~censored) or any(b-a < 2 for a, b in segments):
        flags.append("mass_history_gaps_or_invalid_values")
    return {"raw_mass": mass, "smoothed_mass": smooth, "status": status,
            "positive": positive, "censored": censored, "upper_limit": limits,
            "loss_rate_gyr": rate, "rate_time_gyr": .5*(t[:-1]+t[1:]),
            "rate_is_lower_bound": rate_bound, "quality_flags": flags,
            "valid_sample_count": int(np.count_nonzero(positive)),
            "resolution_supplied": known_resolution}


def _stripping_events(gid, data, tracer, mass_data, cfg):
    t, rate = data["time_gyr"], mass_data["loss_rate_gyr"]
    active = np.isfinite(rate) & (rate > cfg.onset_rate_fraction*cfg.min_loss_rate_gyr)
    changes = np.diff(np.r_[False, active, False].astype(int))
    rows = []
    for number, (a, b) in enumerate(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))):
        peak_index = a + int(np.argmax(rate[a:b]))
        first_mass = mass_data["smoothed_mass"][a]
        is_censored = bool(mass_data["censored"][b])
        last_mass = mass_data["upper_limit"][b] if is_censored else mass_data["smoothed_mass"][b]
        if not np.isfinite(first_mass) or first_mass <= 0 or not np.isfinite(last_mass) or last_mass <= 0:
            continue
        fraction = float(np.clip(1.-last_mass/first_mass, 0., 1.))
        duration = float(t[b]-t[a])
        average_rate = float((np.log(first_mass)-np.log(last_mass))/duration)
        flags = list(mass_data["quality_flags"])
        if is_censored:
            flags.append("fraction_and_rate_are_lower_bounds")
        if a == 0:
            flags.append("onset_left_censored")
        elif not mass_data["positive"][a-1] or t[a]-t[a-1] > data["gap_limit_gyr"]:
            flags.append("onset_after_missing_or_unresolved_mass")
        if b == len(t)-1:
            flags.append("end_right_censored")
        elif not mass_data["positive"][b+1] or t[b+1]-t[b] > data["gap_limit_gyr"]:
            flags.append("end_followed_by_history_gap")
        if b == a+1:
            flags.append("single_interval_loss_rate")
        # A one-output loss that immediately reverses is a dip, not a robust episode.
        rebound = False
        if b+1 < len(t) and t[b+1]-t[b] <= data["gap_limit_gyr"] and mass_data["positive"][b+1]:
            recovered = mass_data["smoothed_mass"][b+1]-last_mass
            rebound = recovered > cfg.rebound_fraction*(first_mass-last_mass)
            if rebound:
                flags.append("transient_rebound")
        significant = (fraction >= cfg.min_fractional_loss and rate[peak_index] >= cfg.min_loss_rate_gyr
                       and duration <= cfg.max_episode_duration_gyr and not rebound)
        onset_uncertainty = .5*(t[a+1]-t[a])
        strength = min(1., fraction/cfg.min_fractional_loss) * min(1., rate[peak_index]/cfg.min_loss_rate_gyr)
        if rebound or duration > cfg.max_episode_duration_gyr:
            strength *= .25
        rows.append({"event_id": f"{gid}:{tracer}:{int(data['iout'][a])}", "galaxy_id": gid,
                     "tracer": tracer,
                     "onset_time_gyr": float(t[a]), "peak_time_gyr": float(.5*(t[peak_index]+t[peak_index+1])),
                     "end_time_gyr": float(t[b]), "onset_snapshot": int(data["iout"][a]),
                     "end_snapshot": int(data["iout"][b]), "peak_loss_rate_gyr": float(rate[peak_index]),
                     "mean_loss_rate_gyr": average_rate, "fractional_loss": fraction,
                     "raw_fractional_loss": float(1.-(last_mass if is_censored else mass_data["raw_mass"][b])/mass_data["raw_mass"][a]),
                     "duration_gyr": duration, "timescale_gyr": 1./average_rate if average_rate > 0 else np.nan,
                     "duration_is_lower_bound": b == len(t)-1 or "end_followed_by_history_gap" in flags,
                     "timescale_is_upper_bound": is_censored, "fraction_is_lower_bound": is_censored,
                     "rate_is_lower_bound": bool(mass_data["rate_is_lower_bound"][peak_index]),
                     "onset_uncertainty_gyr": float(onset_uncertainty),
                     "peak_uncertainty_gyr": float(.5*(t[peak_index+1]-t[peak_index])),
                     "significant": bool(significant), "gas_loss_evidence": float(strength),
                     "quality_flags": sorted(set(flags))})
    return rows


def _stripping_local_cadence(data, time):
    t = data["time_gyr"]
    j = int(np.clip(np.searchsorted(t, time), 0, len(t)-1))
    diffs = np.diff(t[max(0, j-2):min(len(t), j+3)])
    diffs = diffs[diffs <= data["gap_limit_gyr"]]
    return float(np.median(diffs)) if len(diffs) else data["cadence_gyr"]


def _stripping_groups(events, data, cfg):
    """Group onset times for reporting only; never determine a mechanism."""
    groups = []
    for event in sorted((e for e in events if e["significant"]), key=lambda e: (e["onset_time_gyr"], e["tracer"])):
        window = max(cfg.agreement_window_gyr, _stripping_local_cadence(data, event["onset_time_gyr"]))
        target = next((g for g in groups if abs(g[0]["onset_time_gyr"]-event["onset_time_gyr"]) <= window
                       and event["tracer"] not in [e["tracer"] for e in g]), None)
        if target is None:
            groups.append([event])
        else:
            target.append(event)
    return groups


def _stripping_coverage(data, onset, window):
    t = data["time_gyr"]
    samples = (t >= onset-window) & (t <= onset+window)
    dt = np.diff(t)
    lo, hi = max(t[0], onset-window), min(t[-1], onset+window)
    overlap = np.maximum(0., np.minimum(t[1:], hi)-np.maximum(t[:-1], lo))
    if hi <= lo:
        return 0.
    covered = data["shock_interval_covered"] & (dt <= data["gap_limit_gyr"])
    fraction = float(np.sum(overlap*covered)/(2*window))
    # Coverage outside the supplied history is missing, not implicitly covered.
    if np.any(samples & ~data["shock_output_covered"]):
        fraction = min(fraction, float(np.mean(data["shock_output_covered"][samples])))
    return fraction


def _stripping_assess_episode(gid, event, data, cfg):
    """Assess one tracer's significant episode without any tracer voting."""
    onset, peak = event["onset_time_gyr"], event["peak_time_gyr"]
    cadence = _stripping_local_cadence(data, onset)
    cadence = cadence if np.isfinite(cadence) else cfg.max_association_window_gyr*2
    window = max(cfg.window_gyr, cfg.association_cadences*cadence)
    uncertainty = event["onset_uncertainty_gyr"]
    gas_evidence = event["gas_loss_evidence"]
    near = [e for e in data["encounters"] if e["end_time_gyr"] >= onset-window and e["start_time_gyr"] <= onset+uncertainty]
    def encounter_rank(e):
        lag = onset-e["peak_time_gyr"]
        timing = np.exp(-max(abs(lag)-uncertainty, 0.)/window)
        return e["confidence_score"]*timing
    encounter = max(near, key=encounter_rank) if near else None
    shock_time = encounter["peak_time_gyr"] if encounter else np.nan
    lag_shock = onset-shock_time
    shock_timing = float(np.exp(-max(abs(lag_shock)-uncertainty, 0.)/window)) if encounter else 0.
    exposure = encounter["confidence_score"] if encounter and encounter["confirmed"] else 0.
    geometry = encounter["geometry_evidence"] if encounter and encounter["confirmed"] else 0.
    verified = [p for p in data["pericenters"] if p["verified_turning_point"] and p["confidence_score"] >= cfg.pericenter_confidence_cut]
    peri = min(verified, key=lambda p: abs(onset-p["time_gyr"])) if verified else None
    peri_time = peri["time_gyr"] if peri else np.nan
    lag_peri = onset-peri_time
    peri_timing = float(np.exp(-abs(lag_peri)/window)*peri["confidence_score"]) if peri else 0.
    coverage = _stripping_coverage(data, onset, window)
    t, r = data["time_gyr"], data["cluster_distance_kpc"]
    inward_pairs = (t[:-1] >= onset-2*window) & (t[1:] <= onset+uncertainty)
    inward_pairs &= np.isfinite(r[:-1]) & np.isfinite(r[1:]) & (np.diff(t) <= data["gap_limit_gyr"])
    inward_pairs &= data["host_cluster"][:-1] == data["host_cluster"][1:]
    inward = float(np.mean(np.diff(r)[inward_pairs] < 0)) if np.any(inward_pairs) else 0.
    no_shock = min(1., coverage/cfg.min_shock_coverage) if not near else 0.
    # Normalize the remaining geometric/timing/gas weights to sum to one.
    # Shock: exposure .30/.85, timing*exposure .20/.85, gas .20/.85,
    # geometry .15/.85. Infall: pericenter .25/.80, gas .20/.80,
    # inward motion .15/.80, covered absence of shocks .20/.80.
    # Neither pressure nor the number/type of other tracers enters these scores.
    shock_score = (.30*exposure + .20*shock_timing*exposure + .20*gas_evidence + .15*geometry)/.85
    rps_score = (.25*peri_timing + .20*gas_evidence + .15*inward + .20*no_shock)/.80
    reasons = []
    flags = list(event["quality_flags"])
    credible = bool(encounter and encounter["confirmed"] and exposure >= cfg.encounter_score_cut)
    confounded = bool(credible and peri and abs(shock_time-peri_time) <= max(cadence, uncertainty+peri["time_uncertainty_gyr"]))
    # Onset exactly at a sampled encounter remains interval-order ambiguous.
    ordering_unresolved = bool(credible and abs(lag_shock) <= uncertainty and not (encounter["start_time_gyr"] < onset < encounter["end_time_gyr"]))
    if confounded:
        reasons.append("shock_and_pericenter_times_unresolved")
    if ordering_unresolved:
        reasons.append("shock_and_loss_order_unresolved")
    if any(f in flags for f in ("onset_left_censored", "onset_after_missing_or_unresolved_mass")):
        reasons.append("loss_onset_not_observed")
    if "insufficient_resolved_mass_samples" in flags:
        reasons.append("insufficient_resolved_mass_samples")
    if near and not credible:
        reasons.append("merger_shock_association_uncertain")
    if coverage < cfg.min_shock_coverage:
        flags.append("shock_coverage_incomplete")
    if window > cfg.max_association_window_gyr:
        reasons.append("cadence_too_coarse_for_mechanism_ordering")
    local_indices = (t >= onset-window) & (t <= onset+window)
    if np.any((np.diff(t) > data["gap_limit_gyr"]) & (t[:-1] <= onset+window) & (t[1:] >= onset-window)):
        reasons.append("important_history_gap_near_loss")
    if np.any(local_indices & ~np.isfinite(r)):
        reasons.append("cluster_center_gap_near_loss")
    shock_ok = (credible and -uncertainty <= lag_shock <= window
                and shock_score >= cfg.score_cut and not confounded and not ordering_unresolved)
    rps_ok = (peri is not None and abs(lag_peri) <= window
              and coverage >= cfg.min_shock_coverage and not near and rps_score >= cfg.score_cut)
    both_scores = shock_score >= cfg.score_cut and rps_score >= cfg.score_cut
    if both_scores and abs(shock_score-rps_score) < cfg.score_margin:
        reasons.append("both_mechanisms_have_comparable_evidence")
    if (reasons or (shock_ok and rps_ok)):
        category = "mixed_or_ambiguous"
    elif shock_ok and shock_score >= rps_score+cfg.score_margin:
        category = "merger_shock_stripping_candidate"
    elif rps_ok and rps_score >= shock_score+cfg.score_margin:
        category = "ordinary_infall_rps_candidate"
    else:
        category = "mixed_or_ambiguous"
        reasons.append("gas_loss_has_no_distinguishable_stripping_mechanism")
    confidence = "medium" if category != "mixed_or_ambiguous" else "low"
    if encounter:
        flags.extend(encounter["quality_flags"])
    if max(shock_score, rps_score) >= .85 and not flags and category != "mixed_or_ambiguous":
        confidence = "high"
    core = data["merger_epoch"]["core_passage_time_gyr"]
    return {"galaxy_id": gid, "gas_event_ids": [event["event_id"]], "tracer": event["tracer"],
            "onset_time_gyr": onset, "peak_time_gyr": peak, "end_time_gyr": event["end_time_gyr"],
            "fractional_loss": event["fractional_loss"], "peak_loss_rate_gyr": event["peak_loss_rate_gyr"],
            "timescale_gyr": event["timescale_gyr"],
            "shock_encounter_id": encounter["encounter_id"] if encounter else None,
            "associated_front_id": encounter["front_id"] if encounter else None,
            "shock_encounter_time_gyr": shock_time, "pericenter_time_gyr": peri_time,
            "delta_t_shock_gyr": float(lag_shock), "delta_t_peri_gyr": float(lag_peri),
            "delta_t_peak_shock_gyr": float(peak-shock_time), "delta_t_peak_peri_gyr": float(peak-peri_time),
            "association_window_gyr": window, "cadence_gyr": cadence,
            "shock_coverage_fraction": coverage,
            "physical_shock_exposure": bool(encounter and encounter["confirmed"]),
            "credible_merger_shock_encounter": credible,
            "modeled_surface_crossing": bool(encounter and encounter["crossed"]),
            "encounter_geometry_evidence": geometry,
            "merger_shock_origin_evidence": encounter["merger_evidence"] if encounter else np.nan,
            "temporal_coincidence_score": shock_timing, "gas_loss_evidence": gas_evidence,
            "merger_shock_stripping_score": float(shock_score), "ordinary_rps_score": float(rps_score),
            "merger_phase": "unresolved" if not np.isfinite(core) else "before_core_passage" if onset < core else "after_core_passage",
            "delta_t_cluster_core_passage_gyr": float(onset-core),
            "category": category, "confidence": confidence, "ambiguity_reasons": sorted(set(reasons)),
            "quality_flags": sorted(set(flags)), "causal_confirmation": False}


def _stripping_analyze_galaxy(gid, data, cfg, *, store=True):
    tracer_data, events = {}, []
    for tracer in cfg.tracers:
        md = _stripping_mass_series(data, tracer, cfg)
        tracer_data[tracer] = md
        events.extend(_stripping_events(gid, data, tracer, md, cfg))
    if store:
        data["gas_tracers"] = tracer_data
    assessments = [_stripping_assess_episode(gid, event, data, cfg) for event in events if event["significant"]]
    groups = _stripping_groups(events, data, cfg)
    by_event = {a["gas_event_ids"][0]: a for a in assessments}
    # Temporal agreement is descriptive. No timing is averaged before assessing
    # a tracer, and no score or decision below uses these group measurements.
    for group in groups:
        dispersion = float(np.std([e["onset_time_gyr"] for e in group]))
        cadence = _stripping_local_cadence(data, group[0]["onset_time_gyr"])
        scale = max(cfg.agreement_window_gyr, cadence) if np.isfinite(cadence) else cfg.agreement_window_gyr
        for event in group:
            by_event[event["event_id"]].update(
                tracer_temporal_agreement=float(np.exp(-dispersion/scale)),
                onset_dispersion_gyr=dispersion,
                coincident_gas_event_ids=[e["event_id"] for e in group])
    flags = list(data["quality_flags"])
    for md in tracer_data.values():
        flags.extend(md["quality_flags"])
    if assessments:
        main = max(assessments, key=lambda a: (a["fractional_loss"], a["peak_loss_rate_gyr"],
                                             -a["onset_time_gyr"], a["tracer"]))
        row = dict(main)
        row["summary_gas_event_id"] = main["gas_event_ids"][0]
        row["summary_tracer"] = row.pop("tracer")
        row["ambiguity_reasons"] = sorted({r for a in assessments for r in a["ambiguity_reasons"]})
        row["quality_flags"] = sorted({f for a in assessments for f in a["quality_flags"]})
        row["merger_shock_stripping_score"] = max(a["merger_shock_stripping_score"] for a in assessments)
        row["ordinary_rps_score"] = max(a["ordinary_rps_score"] for a in assessments)
        row["confidence"] = min((a["confidence"] for a in assessments), key=("low", "medium", "high").index)
        mechanisms = {a["category"] for a in assessments if a["category"] != "mixed_or_ambiguous"}
        if len(mechanisms) > 1:
            row["category"], row["confidence"] = "mixed_or_ambiguous", "low"
            row["ambiguity_reasons"].append("different_episodes_favor_different_mechanisms")
        if any(a["category"] == "mixed_or_ambiguous" for a in assessments):
            row["category"], row["confidence"] = "mixed_or_ambiguous", "low"
            if mechanisms:
                row["ambiguity_reasons"].append("uncertain_gas_loss_episode")
    else:
        enough = all(md["valid_sample_count"] >= cfg.min_valid_samples
                     and np.mean(md["positive"]) >= cfg.min_mass_history_coverage for md in tracer_data.values())
        enough &= not any(f in data["quality_flags"] for f in ("history_time_gaps", "galaxy_position_gaps"))
        row = {"galaxy_id": gid, "category": "no_strong_stripping" if enough else "mixed_or_ambiguous",
               "confidence": "medium" if enough else "low", "ambiguity_reasons": [] if enough else ["insufficient_gas_history"],
               "quality_flags": [], "merger_shock_stripping_score": 0., "ordinary_rps_score": 0.,
               "onset_time_gyr": np.nan, "peak_time_gyr": np.nan, "end_time_gyr": np.nan,
               "fractional_loss": np.nan, "timescale_gyr": np.nan,
               "shock_encounter_time_gyr": np.nan, "pericenter_time_gyr": np.nan,
               "delta_t_shock_gyr": np.nan, "delta_t_peri_gyr": np.nan,
               "summary_gas_event_id": None, "summary_tracer": None,
               "tracer_temporal_agreement": np.nan, "causal_confirmation": False}
    row["tracers"] = list(cfg.tracers)
    row["gas_event_ids"] = [a["gas_event_ids"][0] for a in assessments]
    row["ambiguity_reasons"] = sorted(set(row["ambiguity_reasons"]))
    row["quality_flags"] = sorted(set(row["quality_flags"]+flags))
    if flags and row["confidence"] == "high":
        row["confidence"] = "medium"
    row["gas_loss_by_tracer"] = {}
    for tracer, md in tracer_data.items():
        selected = [e for e in events if e["tracer"] == tracer]
        selected_assessments = [a for a in assessments if a["tracer"] == tracer]
        categories = {a["category"] for a in selected_assessments}
        sufficient = (md["valid_sample_count"] >= cfg.min_valid_samples
                      and np.mean(md["positive"]) >= cfg.min_mass_history_coverage
                      and not any(f in data["quality_flags"] for f in ("history_time_gaps", "galaxy_position_gaps")))
        tracer_category = (next(iter(categories)) if len(categories) == 1 else "mixed_or_ambiguous"
                           if categories or not sufficient else "no_strong_stripping")
        row["gas_loss_by_tracer"][tracer] = {"status": "measured" if md["valid_sample_count"] >= cfg.min_valid_samples else "insufficient_history",
                                            "valid_sample_count": md["valid_sample_count"],
                                            "event_count": len(selected), "significant_event_count": sum(e["significant"] for e in selected),
                                            "events": selected, "episode_assessments": selected_assessments,
                                            "category": tracer_category, "quality_flags": md["quality_flags"]}
    credible = [e for e in data["encounters"] if e["confirmed"] and e["confidence_score"] >= cfg.encounter_score_cut]
    row["shock_encounter_times_gyr"] = [e["peak_time_gyr"] for e in credible]
    row["pericenter_times_gyr"] = [p["time_gyr"] for p in data["pericenters"] if p["verified_turning_point"]]
    complete = bool(np.all(data["shock_output_covered"]) and len(data["shock_interval_covered"]) and np.all(data["shock_interval_covered"]))
    row["exposure_status"] = "shock_exposed" if credible else "no_detected_exposure_with_coverage" if complete and not data["encounters"] else "unknown_or_uncertain_exposure"
    row["episode_count"] = len(groups)
    row["tracer_episode_count"] = len(assessments)
    return row, events, assessments


def _stripping_sensitivity(gid, data, cfg, baseline):
    variants = [("baseline", None, None, cfg)]
    for name, values in (("min_fractional_loss", cfg.sensitivity_fractions), ("min_loss_rate_gyr", cfg.sensitivity_rates_gyr),
                         ("smoothing_width_gyr", cfg.sensitivity_smoothing_gyr), ("window_gyr", cfg.sensitivity_windows_gyr)):
        for value in values:
            if value != getattr(cfg, name):
                variants.append((f"{name}={value:g}", name, value, replace(cfg, **{name: value})))
    rows = []
    for label, parameter, value, variant in variants:
        row, events, _ = _stripping_analyze_galaxy(gid, data, variant, store=False) if parameter else (baseline, [], [])
        rows.append({"galaxy_id": gid, "variant": label, "parameter": parameter, "value": value,
                     "category": row["category"], "merger_shock_stripping_score": row["merger_shock_stripping_score"],
                     "ordinary_rps_score": row["ordinary_rps_score"], "episode_count": row["episode_count"],
                     "fractional_loss": row["fractional_loss"], "timescale_gyr": row["timescale_gyr"],
                     "delta_t_shock_gyr": row["delta_t_shock_gyr"], "delta_t_peri_gyr": row["delta_t_peri_gyr"]})
    return rows


def _stripping_value(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _stripping_frame(product, snapshot, box, cfg):
    """Compact saved merger-front membership; never run a detector."""
    if product is None:
        return None
    members = product.get("members")
    result = product.get("result")
    time_gyr, epoch = product.get("time_gyr"), None
    if members is None:
        raise ValueError("supply independently assessed merger members; generic shock_front_catalog outputs have no merger evidence")
    fronts = {}
    for member in members:
        if int(member["iout"]) != snapshot:
            raise ValueError("member snapshot does not match its product key")
        assessment = member["assessment"]
        if assessment["classification"] not in ("candidate", "uncertain") or assessment["evidence"] < cfg.min_front_evidence:
            continue
        track = assessment.get("track_origin") or assessment.get("front_id")
        if not isinstance(track, str) or not track or track in fronts:
            raise ValueError("plausible merger fronts require unique front_id or independently supplied track_origin values")
        factor = _LENGTH_TO_KPC.get(member.get("position_unit"))
        if factor is None:
            raise ValueError("shock members require explicit physical km/kpc/Mpc position_unit")
        if member.get("dissipation_units") != {"flux": "erg/s/kpc2", "total": "erg/s", "area": "kpc2"}:
            raise ValueError("supply saved flux/total/area and their ShockFinder dissipation units")
        ids = np.asarray(member["shock_id"])
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
            raise ValueError("shock_id must contain unique existing nonnegative center indices")
        n = len(ids)
        if not n:
            continue
        pos = _stripping_column(member, "pos", n, shape=(n, 3))*factor
        dx = _stripping_column(member, "dx", n)*factor
        normal = _stripping_column(member, "normal", n, shape=(n, 3))
        norm = np.linalg.norm(normal, axis=1)
        mach, flux, total, area = [_stripping_column(member, key, n) for key in ("mach", "flux", "total", "area")]
        if (not np.all(np.isfinite(pos)) or not np.all(np.isfinite(dx) & (dx > 0))
                or not np.all(np.isfinite(norm) & (norm > 0)) or not np.all(np.isfinite(mach) & (mach > 1))
                or not np.all(np.isfinite(flux) & (flux >= 0)) or not np.all(np.isfinite(total) & (total >= 0))
                or not np.all(np.isfinite(area) & (area > 0))):
            raise ValueError("merger member cells must have passed saved-field validity selection")
        normal /= norm[:, None]
        oriented = np.zeros(n, bool)
        flags = list(assessment.get("quality_flags", ()))
        upstream_pos, downstream_pos = member.get("upstream_pos"), member.get("downstream_pos")
        if ((upstream_pos is None or downstream_pos is None) and result is not None
                and _stripping_value(result, "pos") is not None
                and "upstream_index" in member and "downstream_index" in member):
            if _stripping_value(result, "position_unit") != member["position_unit"]:
                raise ValueError("member and result endpoint position units disagree")
            up, down = np.asarray(member["upstream_index"]), np.asarray(member["downstream_index"])
            source_pos = np.asarray(_stripping_value(result, "pos"))
            valid = (up >= 0) & (down >= 0) & (up < len(source_pos)) & (down < len(source_pos))
            upstream_pos, downstream_pos = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
            upstream_pos[valid], downstream_pos[valid] = source_pos[up[valid]], source_pos[down[valid]]
        if upstream_pos is not None and downstream_pos is not None:
            up, down = np.asarray(upstream_pos, float)*factor, np.asarray(downstream_pos, float)*factor
            if up.shape != (n, 3) or down.shape != (n, 3):
                raise ValueError("endpoint positions must have shape (N,3) in member position_unit")
            delta = _stripping_delta(down-up, box)
            endpoint_norm = np.linalg.norm(delta, axis=1)
            projection = np.sum(delta*normal, axis=1)
            oriented = np.isfinite(endpoint_norm) & (endpoint_norm > 0) & (np.abs(projection) >= cfg.normal_cosine_min*endpoint_norm)
            normal[oriented] *= np.sign(projection[oriented, None])
        if not np.all(oriented):
            flags.append("upstream_downstream_orientation_unavailable")
        zone = _stripping_column(member, "zone_width", n, default=0.)
        if not np.all(np.isfinite(zone) & (zone >= 0)):
            raise ValueError("zone_width must be a nonnegative cell count")
        if "zone_width" not in member:
            flags.append("zone_width_not_saved")
        disk_radius = np.sqrt(3.)*.5*dx
        half = .5*dx*np.sum(np.abs(normal), axis=1)+cfg.zone_width_factor*zone*dx
        local = _stripping_delta(pos-pos[0], box)
        center = pos[0]+np.average(local, axis=0, weights=area)
        front = {"front_id": track, "shock_id": ids.copy(), "pos": pos, "dx": dx,
                 "normal": normal, "oriented": oriented, "radius": disk_radius, "half": half,
                 "mach": mach, "flux": flux, "total": total, "area": area,
                 "center": center, "evidence": float(assessment["evidence"]),
                 "classification": assessment["classification"], "quality_flags": flags,
                 "representative_shock_id": int(assessment["shock_id"]), "box": box, "buckets": []}
        front["level"] = _stripping_column(member, "level", n)
        if not 0 <= front["evidence"] <= 1:
            raise ValueError("merger evidence must be an index in [0,1]")
        levels = np.floor(np.log2(dx/np.min(dx))+1.e-8).astype(int)
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        for level in np.unique(levels):
            ii = np.flatnonzero(levels == level)
            xyz = pos[ii] if box is None else pos[ii] % box
            tree = cKDTree(xyz, boxsize=box) if cKDTree is not None else None
            front["buckets"].append((ii, tree, float(np.max(np.hypot(disk_radius[ii], half[ii])))))
        fronts[track] = front
    return {"fronts": fronts, "complete": bool(product.get("complete", False)),
            "time_gyr": time_gyr, "epoch": epoch,
            "covered_galaxy_ids": product.get("covered_galaxy_ids")}


def _stripping_ball(front, point, radius):
    rows = []
    box = front["box"]
    for ii, tree, support in front["buckets"]:
        if tree is not None:
            selected = tree.query_ball_point(point if box is None else point % box, radius+support)
            rows.extend(ii[selected])
        else:
            delta = _stripping_delta(front["pos"][ii]-point, box)
            rows.extend(ii[np.linalg.norm(delta, axis=1) <= radius+support])
    return np.asarray(rows, dtype=int)


def _stripping_nearest_center(front, point):
    best, index = np.inf, -1
    for ii, tree, _ in front["buckets"]:
        if tree is not None:
            distance, j = tree.query(point if front["box"] is None else point % front["box"])
        else:
            distances = np.linalg.norm(_stripping_delta(front["pos"][ii]-point, front["box"]), axis=1)
            j, distance = int(np.argmin(distances)), float(np.min(distances))
        if distance < best:
            best, index = float(distance), int(ii[j])
    return index, best


def _stripping_query_front(front, point, gas_radius, velocity):
    _, nearest_center_distance = _stripping_nearest_center(front, point)
    rows = _stripping_ball(front, point, max(nearest_center_distance, gas_radius))
    delta = _stripping_delta(point-front["pos"][rows], front["box"])
    signed = np.sum(delta*front["normal"][rows], axis=1)
    tangent = np.linalg.norm(delta-signed[:, None]*front["normal"][rows], axis=1)
    distance = np.hypot(signed, np.maximum(tangent-front["radius"][rows], 0.))
    j = int(np.argmin(distance))
    nearest = int(rows[j])
    # Gas sphere vs finite shock-zone cylinder (explicit numerical-zone proxy).
    zone_distance = np.hypot(np.maximum(np.abs(signed)-front["half"][rows], 0.),
                             np.maximum(tangent-front["radius"][rows], 0.))
    local = rows[zone_distance <= gas_radius]
    near = len(local) > 0
    sample_rows = local if near else np.array([], dtype=int)
    mean_normal = np.average(front["normal"][local], axis=0, weights=front["area"][local]) if near else front["normal"][nearest].copy()
    coherence = float(np.linalg.norm(mean_normal))
    normal = mean_normal/coherence if coherence > 0 else np.full(3, np.nan)
    speed = np.linalg.norm(velocity)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(velocity, normal)/speed, -1., 1.)))) if np.isfinite(speed) and speed > 0 and np.all(np.isfinite(normal)) else np.nan
    side = "unknown"
    if front["oriented"][nearest] and tangent[j] <= front["radius"][nearest]+gas_radius:
        side = "straddling" if abs(signed[j]) <= front["half"][nearest]+gas_radius else "upstream" if signed[j] < 0 else "downstream"
    def statistic(key, fn):
        return float(fn(front[key][sample_rows])) if len(sample_rows) else np.nan
    return {"front_id": front["front_id"], "shock_id": int(front["shock_id"][nearest]),
            "surface_distance_kpc": float(distance[j]),
            "cell_distance_kpc": float(np.linalg.norm(np.maximum(np.abs(delta[j])-.5*front["dx"][nearest], 0.))),
            "signed_distance_kpc": float(signed[j]), "normal": normal,
            "cell_size_kpc": float(front["dx"][nearest]), "amr_level": float(front["level"][nearest]),
            "normal_coherence": coherence, "velocity_normal_angle_deg": angle, "side": side,
            "near": near, "local_cell_count": len(sample_rows), "nearest_mach": float(front["mach"][nearest]),
            "nearest_flux_erg_s_kpc2": float(front["flux"][nearest]),
            "local_mach_median": statistic("mach", np.median), "local_mach_min": statistic("mach", np.min),
            "local_mach_max": statistic("mach", np.max), "local_flux_median_erg_s_kpc2": statistic("flux", np.median),
            "local_dissipation_total_erg_s": statistic("total", np.sum),
            "local_dissipation_median_erg_s": statistic("total", np.median),
            "merger_evidence": front["evidence"], "merger_classification": front["classification"],
            "quality_flags": front["quality_flags"]}


def _stripping_clip_cylinder(start, delta, normal, radius, half):
    s0, ds = float(np.dot(start, normal)), float(np.dot(delta, normal))
    lo, hi = 0., 1.
    if abs(ds) < 1.e-12:
        if abs(s0) > half:
            return None
    else:
        a, b = sorted(((-half-s0)/ds, (half-s0)/ds))
        lo, hi = max(lo, a), min(hi, b)
    tangent, tangent_delta = start-s0*normal, delta-ds*normal
    aa, bb, cc = float(tangent_delta@tangent_delta), float(2*tangent@tangent_delta), float(tangent@tangent-radius**2)
    if aa > 1.e-24:
        discriminant = bb**2-4*aa*cc
        if discriminant < 0:
            return None
        roots = ((-bb-np.sqrt(discriminant))/(2*aa), (-bb+np.sqrt(discriminant))/(2*aa))
        lo, hi = max(lo, roots[0]), min(hi, roots[1])
    elif cc > 0:
        return None
    if hi < lo:
        return None
    crossing = -s0/ds if ds != 0 else np.nan
    crossing = crossing if lo <= crossing <= hi and 0 < crossing <= 1 else np.nan
    return lo, hi, crossing


def _stripping_pair_front(data, left_index, right_index, previous, current, cfg):
    p0, p1 = data["pos_kpc"][[left_index, right_index]]
    t0, t1 = data["time_gyr"][[left_index, right_index]]
    dt = t1-t0
    radius_valid = data["radius_valid"][[left_index, right_index]].all()
    gas_radius = float(np.min(data["radius_kpc"][[left_index, right_index]])) if radius_valid else 0.
    shift = _stripping_delta(current["center"]-previous["center"], current["box"])
    galaxy_delta = _stripping_delta(p1-p0, current["box"])
    relative_delta = galaxy_delta-shift
    padding = cfg.patch_match_cells*.5*(np.max(previous["dx"])+np.max(current["dx"]))
    candidates = _stripping_ball(previous, p0+.5*relative_delta, .5*np.linalg.norm(relative_delta)+gas_radius+padding)
    hits, complete = [], True
    for a in candidates:
        predicted_start = _stripping_delta(p0-previous["pos"][a], previous["box"])
        coarse = _stripping_clip_cylinder(predicted_start, relative_delta, previous["normal"][a],
                                         previous["radius"][a]+gas_radius+padding, previous["half"][a]+gas_radius+padding)
        if coarse is None:
            continue
        expected = previous["pos"][a]+shift
        b, residual = _stripping_nearest_center(current, expected)
        mean_dx = .5*(previous["dx"][a]+current["dx"][b])
        if residual > cfg.patch_match_cells*mean_dx:
            complete = False
            continue
        n0, n1 = previous["normal"][a], current["normal"][b].copy()
        alignment = float(n0@n1)
        oriented = previous["oriented"][a] and current["oriented"][b]
        if abs(alignment) < cfg.normal_cosine_min or (oriented and alignment < 0):
            complete = False
            continue
        if alignment < 0:
            n1 *= -1
        normal = n0+n1
        normal /= np.linalg.norm(normal)
        end = _stripping_delta(p1-current["pos"][b], current["box"])
        # Choose the adjacent periodic image of the SAME tracked local patch.
        end = predicted_start+_stripping_delta(end-predicted_start, current["box"])
        delta = end-predicted_start
        relative_speed = np.linalg.norm(delta)*KPC_IN_KM/(dt*_GYR_IN_S)
        patch_motion = np.linalg.norm(_stripping_delta(current["pos"][b]-previous["pos"][a], current["box"]))
        patch_speed = patch_motion*KPC_IN_KM/(dt*_GYR_IN_S)
        if relative_speed > cfg.max_motion_km_s or patch_speed > cfg.max_motion_km_s:
            complete = False
            continue
        radius = min(previous["radius"][a], current["radius"][b])+gas_radius
        half = min(previous["half"][a], current["half"][b])+gas_radius
        intersection = _stripping_clip_cylinder(predicted_start, delta, normal, radius, half)
        if intersection is None:
            continue
        lo, hi, crossing = intersection
        crosses = bool(np.isfinite(crossing))
        phase = float(crossing) if crosses else float(.5*(lo+hi))
        candidate = previous["classification"] == current["classification"] == "candidate"
        origin_evidence = min(previous["evidence"], current["evidence"])
        geometry = .95 if crosses else .65
        score = geometry*np.sqrt(origin_evidence*abs(alignment))*np.exp(-.15*residual/mean_dx)
        if not candidate:
            score *= .65
        if not radius_valid:
            score *= .6
        motion_norm = np.linalg.norm(delta)
        angle = float(np.degrees(np.arccos(np.clip(delta@normal/motion_norm, -1., 1.)))) if motion_norm > 0 else np.nan
        flags = previous["quality_flags"]+current["quality_flags"]+[
            "moving_planar_patch_model", "linear_motion_between_outputs", "AMR_patch_footprint_approximation"]
        if not oriented:
            flags.append("crossing_side_orientation_unknown")
        if not radius_valid:
            flags.append("gas_radius_missing_or_invalid")
        hits.append({"front_id": current["front_id"], "start_time_gyr": float(t0+lo*dt),
                     "peak_time_gyr": float(t0+phase*dt), "end_time_gyr": float(t0+hi*dt),
                     "crossing_times_gyr": [float(t0+crossing*dt)] if crosses else [],
                     "crossed": crosses, "confirmed": True, "geometry_evidence": geometry,
                     "geometry_status": "moving_surface_crossing" if crosses else "consistent_zone_exposure",
                     "confidence_score": float(score), "merger_evidence": origin_evidence,
                     "merger_classification": "candidate" if candidate else "uncertain",
                     "snapshots": [int(data["iout"][left_index]), int(data["iout"][right_index])],
                     "shock_ids": [(int(data["iout"][left_index]), int(previous["shock_id"][a])),
                                   (int(data["iout"][right_index]), int(current["shock_id"][b]))],
                     "peak_mach": float(max(previous["mach"][a], current["mach"][b])),
                     "flux_median_erg_s_kpc2": float(.5*(previous["flux"][a]+current["flux"][b])),
                     "dissipation_median_erg_s": float(.5*(previous["total"][a]+current["total"][b])),
                     "normal": normal.tolist(), "relative_motion_normal_angle_deg": angle,
                     "relative_normal_speed_km_s": float(delta@normal*KPC_IN_KM/(dt*_GYR_IN_S)),
                     "time_uncertainty_gyr": float(.5*dt), "quality_flags": sorted(set(flags))})
    return hits, complete


def _stripping_encounter_union(gid, hits, observations, data):
    groups = []
    for hit in sorted(hits, key=lambda h: (h["front_id"], h["start_time_gyr"])):
        group = next((g for g in reversed(groups) if g[0]["front_id"] == hit["front_id"]
                      and hit["start_time_gyr"] <= max(e["end_time_gyr"] for e in g)+1.e-9), None)
        if group is None:
            groups.append([hit])
        else:
            group.append(hit)
    encounters = []
    for number, group in enumerate(groups):
        main = max(group, key=lambda h: (h["crossed"], h["confidence_score"]))
        row = dict(main)
        row["galaxy_id"], row["encounter_id"] = gid, f"{gid}:encounter:{number}"
        row["start_time_gyr"] = min(h["start_time_gyr"] for h in group)
        row["end_time_gyr"] = max(h["end_time_gyr"] for h in group)
        row["duration_gyr"] = row["end_time_gyr"]-row["start_time_gyr"]
        row["snapshots"] = sorted({s for h in group for s in h["snapshots"]})
        row["shock_ids"] = sorted({tuple(s) for h in group for s in h["shock_ids"]})
        row["crossing_times_gyr"] = sorted({round(t, 12) for h in group for t in h["crossing_times_gyr"]})
        row["crossed"] = bool(row["crossing_times_gyr"])
        row["quality_flags"] = sorted({f for h in group for f in h["quality_flags"]})
        row["confidence"] = "high" if row["confidence_score"] >= .8 else "medium" if row["confidence_score"] >= .5 else "low"
        encounters.append(row)
    # Keep isolated hits explicitly, rather than upgrading proximity to a crossing.
    for sample in observations:
        if not sample["near"]:
            continue
        if any(e["front_id"] == sample["front_id"] and sample["snapshot"] in e["snapshots"] for e in encounters):
            continue
        time = sample["time_gyr"]
        encounters.append({"encounter_id": f"{gid}:proximity:{sample['snapshot']}:{sample['front_id']}",
                           "galaxy_id": gid, "front_id": sample["front_id"],
                           "start_time_gyr": time, "peak_time_gyr": time, "end_time_gyr": time,
                           "duration_gyr": 0., "crossing_times_gyr": [], "crossed": False,
                           "confirmed": False, "confidence_score": .3*sample["merger_evidence"],
                           "confidence": "low", "geometry_evidence": .2, "geometry_status": "single_output_proximity",
                           "merger_evidence": sample["merger_evidence"], "merger_classification": sample["merger_classification"],
                           "snapshots": [sample["snapshot"]], "shock_ids": [(sample["snapshot"], sample["shock_id"])],
                           "peak_mach": sample["local_mach_max"], "normal": sample["normal"].tolist(),
                           "flux_median_erg_s_kpc2": sample["local_flux_median_erg_s_kpc2"],
                           "dissipation_median_erg_s": sample["local_dissipation_median_erg_s"],
                           "time_uncertainty_gyr": data["cadence_gyr"],
                           "quality_flags": sorted(set(sample["quality_flags"]+["single_output_proximity_unconfirmed"]))})
    return sorted(encounters, key=lambda e: e["peak_time_gyr"])


def _stripping_associate_shocks(series, merger_shocks, cfg):
    schedule, lookup = {}, {}
    hits, observations = {gid: [] for gid in series}, {gid: [] for gid in series}
    provenance = {}
    for gid, data in series.items():
        n = len(data["iout"])
        for name in ("shock_distance_kpc", "shock_cell_distance_kpc", "shock_cell_size_kpc", "shock_amr_level",
                     "shock_mach", "shock_flux_erg_s_kpc2", "shock_local_mach", "shock_local_mach_min", "shock_local_mach_max",
                     "shock_local_dissipation_erg_s", "shock_local_dissipation_median_erg_s", "shock_local_flux_erg_s_kpc2",
                     "shock_normal_coherence", "velocity_normal_angle_deg"):
            data[name] = np.full(n, np.nan)
        data["shock_id"] = np.full(n, -1, dtype=np.int64)
        data["shock_front_id"] = np.full(n, None, dtype=object)
        data["shock_side"] = np.full(n, "unknown", dtype=object)
        data["shock_normal"] = np.full((n, 3), np.nan)
        data["shock_output_covered"] = np.zeros(n, bool)
        data["shock_interval_covered"] = np.zeros(max(n-1, 0), bool)
        lookup[gid] = {int(s): j for j, s in enumerate(data["iout"])}
        for j, snapshot in enumerate(data["iout"]):
            s = int(snapshot)
            box = None if data["box_kpc"] is None else data["box_kpc"][j]
            if s in schedule:
                time, old_box = schedule[s]
                if abs(time-data["time_gyr"][j]) > 1.e-6 or (old_box is None) != (box is None) or (box is not None and not np.allclose(old_box, box)):
                    raise ValueError("galaxies must share time and periodic-box metadata at each snapshot")
            schedule[s] = (data["time_gyr"][j], box)
    previous, previous_snapshot = None, None
    for snapshot in sorted(schedule, key=lambda s: schedule[s][0]):
        product = merger_shocks(snapshot) if callable(merger_shocks) else merger_shocks.get(snapshot)
        if product is not None:
            provenance[snapshot] = {"supplied": product.get("provenance"),
                                    "complete": bool(product.get("complete", False))}
        frame = _stripping_frame(product, snapshot, schedule[snapshot][1], cfg)
        if frame is not None and frame["time_gyr"] is not None and not np.isclose(frame["time_gyr"], schedule[snapshot][0], atol=1.e-6, rtol=0.):
            raise ValueError("merger catalog and galaxy cosmic times disagree for the same snapshot")
        # Only compact previous/current cells are retained, supporting loaders.
        product = None
        for gid, data in series.items():
            j = lookup[gid].get(snapshot)
            if j is None:
                continue
            if frame is not None and frame["epoch"] and frame["epoch"].get("verified"):
                supplied_core = frame["epoch"]["time_gyr"]
                measured_core = data["merger_epoch"]["core_passage_time_gyr"]
                if np.isfinite(measured_core) and not np.isclose(measured_core, supplied_core, atol=1.e-6, rtol=0.):
                    raise ValueError("merger catalog and cluster histories disagree on core passage")
                data["merger_epoch"] = {"core_passage_time_gyr": supplied_core, "verified": True,
                                        "source": "independent merger-catalog epoch verification"}
            covered = bool(frame and frame["complete"] and (frame["covered_galaxy_ids"] is None or gid in frame["covered_galaxy_ids"]))
            data["shock_output_covered"][j] = covered
            position_valid = np.all(np.isfinite(data["pos_kpc"][j]))
            if frame is not None and position_valid:
                gas_radius = data["radius_kpc"][j] if data["radius_valid"][j] else 0.
                samples = [_stripping_query_front(f, data["pos_kpc"][j], gas_radius, data["velocity_km_s"][j]) for f in frame["fronts"].values()]
                if samples:
                    nearest = min(samples, key=lambda s: s["surface_distance_kpc"])
                    data["shock_distance_kpc"][j] = nearest["surface_distance_kpc"]
                    data["shock_cell_distance_kpc"][j] = nearest["cell_distance_kpc"]
                    data["shock_cell_size_kpc"][j], data["shock_amr_level"][j] = nearest["cell_size_kpc"], nearest["amr_level"]
                    data["shock_mach"][j], data["shock_flux_erg_s_kpc2"][j] = nearest["nearest_mach"], nearest["nearest_flux_erg_s_kpc2"]
                    data["shock_local_mach"][j], data["shock_local_dissipation_erg_s"][j] = nearest["local_mach_median"], nearest["local_dissipation_total_erg_s"]
                    data["shock_local_mach_min"][j], data["shock_local_mach_max"][j] = nearest["local_mach_min"], nearest["local_mach_max"]
                    data["shock_local_dissipation_median_erg_s"][j] = nearest["local_dissipation_median_erg_s"]
                    data["shock_local_flux_erg_s_kpc2"][j], data["shock_normal_coherence"][j] = nearest["local_flux_median_erg_s_kpc2"], nearest["normal_coherence"]
                    data["shock_id"][j], data["shock_front_id"][j] = nearest["shock_id"], nearest["front_id"]
                    data["shock_normal"][j], data["shock_side"][j] = nearest["normal"], nearest["side"]
                    data["velocity_normal_angle_deg"][j] = nearest["velocity_normal_angle_deg"]
                    for sample in samples:
                        sample.update(snapshot=snapshot, time_gyr=float(data["time_gyr"][j]))
                        observations[gid].append(sample)
                elif covered:
                    data["shock_distance_kpc"][j] = np.inf
            if j > 0 and previous_snapshot == int(data["iout"][j-1]) and previous is not None and frame is not None:
                dt = data["time_gyr"][j]-data["time_gyr"][j-1]
                trajectory_valid = np.all(np.isfinite(data["pos_kpc"][[j-1, j]])) and dt <= data["gap_limit_gyr"]
                if trajectory_valid:
                    speed = np.linalg.norm(_stripping_delta(data["pos_kpc"][j]-data["pos_kpc"][j-1], schedule[snapshot][1]))*KPC_IN_KM/(dt*_GYR_IN_S)
                    trajectory_valid &= speed <= cfg.max_motion_km_s
                    if not trajectory_valid:
                        data["quality_flags"].append("galaxy_trajectory_speed_exceeds_limit")
                tracked = previous["fronts"].keys() == frame["fronts"].keys()
                interval_complete = trajectory_valid and tracked and covered and data["shock_output_covered"][j-1]
                if trajectory_valid:
                    for track in previous["fronts"].keys() & frame["fronts"].keys():
                        segments, patch_complete = _stripping_pair_front(data, j-1, j, previous["fronts"][track], frame["fronts"][track], cfg)
                        hits[gid].extend(segments)
                        interval_complete &= patch_complete
                data["shock_interval_covered"][j-1] = interval_complete
        previous, previous_snapshot = frame, snapshot
    all_encounters = []
    for gid, data in series.items():
        data["shock_observations"] = observations[gid]
        data["encounters"] = _stripping_encounter_union(gid, hits[gid], observations[gid], data)
        all_encounters.extend(data["encounters"])
        if not data["shock_output_covered"].all():
            data["quality_flags"].append("shock_output_coverage_missing")
    return all_encounters, provenance


def _stripping_population(analysis):
    rows = analysis["classifications"]
    groups = {}
    for status in ("shock_exposed", "no_detected_exposure_with_coverage", "unknown_or_uncertain_exposure"):
        selected = [r for r in rows if r["exposure_status"] == status]
        losses = [r for r in selected if r["episode_count"] > 0]
        groups[status] = {"galaxy_count": len(selected), "galaxies_with_significant_loss": len(losses),
                          "significant_loss_fraction": len(losses)/len(selected) if selected else np.nan,
                          "median_loss_fraction": float(np.median([r["fractional_loss"] for r in losses])) if losses else np.nan,
                          "median_timescale_gyr": float(np.median([r["timescale_gyr"] for r in losses])) if losses else np.nan}
    return {"category_counts": {c: sum(r["category"] == c for r in rows) for c in STRIPPING_CATEGORIES},
            "exposure_comparison": groups,
            "comparisons_are_descriptive": True, "causal_confirmation": False,
            "sensitivity_category_counts": {
                v: {c: sum(r["variant"] == v and r["category"] == c for r in analysis["sensitivity"]) for c in STRIPPING_CATEGORIES}
                for v in sorted({r["variant"] for r in analysis["sensitivity"]})}}


def plot_galaxy_stripping(analysis, galaxy_id):
    """Return a Matplotlib figure for one history; do not modify inputs."""
    import matplotlib.pyplot as plt
    data = analysis["diagnostic_series"][galaxy_id]
    t = data["time_gyr"]
    fig, axes = plt.subplots(7, 1, figsize=(11, 17), sharex=True, constrained_layout=True)
    row = next(r for r in analysis["classifications"] if r["galaxy_id"] == galaxy_id)
    fig.suptitle(f"Galaxy {galaxy_id}: {row['category']} ({row['confidence']})\nEvidence for a mechanism; causal confirmation = False")
    axes[0].plot(t, data["primary_distance_kpc"], label="primary")
    axes[0].plot(t, data["secondary_distance_kpc"], label="secondary")
    axes[0].plot(t, data["cluster_distance_kpc"], "k--", label="adopted host")
    axes[0].set_ylabel("Cluster distance [kpc]")
    axes[0].legend(loc="best", ncol=3)
    palette = plt.get_cmap("tab10").colors
    for number, (tracer, md) in enumerate(data["gas_tracers"].items()):
        positive = md["positive"]
        scale = md["raw_mass"][np.flatnonzero(positive)[0]] if np.any(positive) else np.nan
        if np.isfinite(scale):
            color = palette[number % len(palette)]
            axes[1].plot(t, np.where(positive, md["raw_mass"]/scale, np.nan), ".", color=color, alpha=.5)
            axes[1].plot(t, md["smoothed_mass"]/scale, color=color, label=tracer)
            censored = md["censored"]
            axes[1].scatter(t[censored], md["upper_limit"][censored]/scale, color=color, marker="v")
            axes[2].plot(md["rate_time_gyr"], md["loss_rate_gyr"], color=color)
    axes[1].set_ylabel("Gas mass / first valid mass")
    if axes[1].lines:
        axes[1].legend(ncol=2, fontsize=8)
    axes[2].axhline(analysis["configuration"]["min_loss_rate_gyr"], color="k", ls="--", label="rapid-loss threshold")
    axes[2].set_ylabel("−d ln M / dt [Gyr⁻¹]")
    axes[2].legend(fontsize=8)
    # Pressure is read only for this diagnostic, after all decisions are made.
    p = _stripping_column(data["raw"], "P_ram", len(t))[data["order"]]
    axes[3].plot(t, np.where(np.isfinite(p) & (p > 0), p, np.nan), color="purple")
    if np.any(np.isfinite(p) & (p > 0)):
        axes[3].set_yscale("log")
    axes[3].set_ylabel("P_ram [input unit]\nDiagnostic only")
    d = data["shock_distance_kpc"]
    axes[4].plot(t, np.where(np.isfinite(d), d, np.nan), label="nearest plausible merger surface")
    axes[4].plot(t, np.where(data["radius_valid"], data["radius_kpc"], np.nan), ls="--", label="gas aperture")
    axes[4].set_ylabel("Shock distance [kpc]")
    axes[4].legend(fontsize=8)
    axes[5].plot(t, data["shock_mach"], label="nearest-cell Mach", color="teal")
    axes[5].plot(t, data["shock_local_mach"], "o", label="local Mach during overlap", color="darkgreen")
    axes[5].set_ylabel("Mach")
    axes[5].legend(fontsize=8)
    axes[6].plot(t, data["shock_flux_erg_s_kpc2"], color="orange", label="nearest-cell flux")
    axes[6].set_ylabel("Flux [erg s⁻¹ kpc⁻²]")
    total_axis = axes[6].twinx()
    total_axis.plot(t, data["shock_local_dissipation_erg_s"], color="red", ls=":", marker="o", ms=3, label="local cell-integrated rate")
    total_axis.set_ylabel("Local rate [erg s⁻¹]", color="red")
    for ax in axes:
        for peri in data["pericenters"]:
            if peri["verified_turning_point"]:
                ax.axvline(peri["time_gyr"], color="black", ls=":", alpha=.5)
        for enc in data["encounters"]:
            if enc["confirmed"]:
                ax.axvspan(enc["start_time_gyr"], enc["end_time_gyr"], color="cyan", alpha=.15)
                ax.axvline(enc["peak_time_gyr"], color="teal", ls="--", alpha=.4)
            else:
                ax.axvline(enc["peak_time_gyr"], color="grey", ls=":", alpha=.3)
        for episode in analysis["episode_assessments"]:
            if episode["galaxy_id"] == galaxy_id:
                ax.axvspan(episode["onset_time_gyr"], episode["end_time_gyr"], color="tomato", alpha=.12)
                ax.axvline(episode["peak_time_gyr"], color="red", alpha=.3)
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Cosmic time [Gyr]; cyan: modeled encounter, red: gas loss, dotted black: pericenter")
    return fig, axes


def plot_stripping_population(analysis):
    """Descriptive comparisons and the actual one-option-at-a-time sweep."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax = axes.ravel()
    rows = analysis["classifications"]
    colors = dict(zip(STRIPPING_CATEGORIES, ("teal", "darkorange", "grey", "royalblue")))
    groups = analysis["population"]["exposure_comparison"]
    labels = ("Exposed", "Covered; none detected", "Unknown exposure")
    fractions = [g["significant_loss_fraction"] for g in groups.values()]
    ax[0].bar(range(3), fractions, color=("teal", "orange", "grey"))
    ax[0].set_xticks(range(3), labels, rotation=15, ha="right")
    ax[0].set_xlim(-.6, 2.6)
    ax[0].set_ylabel("Fraction with significant gas loss")
    for j, g in enumerate(groups.values()):
        ax[0].text(j, 1., f"N={g['galaxy_count']}", ha="center")
    ax[0].set_ylim(0, 1.15)
    samples, ticklabels = [], []
    for c in STRIPPING_CATEGORIES:
        values = [r["timescale_gyr"] for r in rows if r["category"] == c and np.isfinite(r["timescale_gyr"])]
        if values:
            samples.append(values)
            ticklabels.append(c.replace("_candidate", "").replace("_", " "))
    if samples:
        ax[1].boxplot(samples)
        ax[1].set_xticks(range(1, len(ticklabels)+1), ticklabels, rotation=20, ha="right")
    ax[1].set_ylabel("Gas-loss timescale [Gyr]")
    for j, key in ((2, "delta_t_shock_gyr"), (3, "delta_t_peri_gyr")):
        for c in STRIPPING_CATEGORIES:
            measured = [r for r in rows if r["category"] == c and np.isfinite(r[key]) and np.isfinite(r["fractional_loss"])]
            ax[j].scatter([r[key] for r in measured], [r["fractional_loss"] for r in measured], color=colors[c], label=c)
        ax[j].axvline(0, color="k", ls=":")
        ax[j].set_xlabel(key.replace("_", " "))
        ax[j].set_ylabel("Main episode fractional gas loss")
    counts = analysis["population"]["sensitivity_category_counts"]
    variants = list(counts)
    bottom = np.zeros(len(variants))
    for c in STRIPPING_CATEGORIES:
        values = np.array([counts[v][c] for v in variants])
        ax[4].bar(range(len(variants)), values, bottom=bottom, color=colors[c], label=c)
        bottom += values
    ax[4].set_xticks(range(len(variants)), variants, rotation=60, ha="right", fontsize=7)
    ax[4].set_ylabel("Galaxy count by sensitivity variant")
    stability = [r["classification_stability"] for r in rows]
    ax[5].hist(stability, bins=np.linspace(0, 1, 11), color="slategrey")
    ax[5].set_xlabel("Fraction of variants agreeing with baseline")
    ax[5].set_ylabel("Galaxy count")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], label=c.replace("_", " ")) for c in STRIPPING_CATEGORIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=2, fontsize=8)
    fig.suptitle("Descriptive population comparisons\nExposure and time coincidence do not establish causality", y=1.10)
    return fig, axes


def save_galaxy_stripping_analysis(analysis, output_dir, *, make_plots=True):
    """Save reproducible tables/parameters and plots; pickle retains all inputs."""
    import csv
    import json
    import pickle
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return plain(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        return value
    paths = {}
    for name in ("gas_loss_events", "classifications", "shock_encounters", "pericenters", "episode_assessments", "sensitivity"):
        path = output_dir/f"{name}.csv"
        rows = analysis[name]
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                values = {k: json.dumps(plain(v), allow_nan=False) if isinstance(v, (dict, list, tuple, np.ndarray)) else plain(v) for k, v in row.items()}
                writer.writerow(values)
        paths[name] = str(path.resolve())
    metadata = {"schema_version": analysis["schema_version"], "configuration": analysis["configuration"],
                "population": analysis["population"], "interpretation": analysis["interpretation"],
                "merger_epochs": {str(g): d["merger_epoch"] for g, d in analysis["diagnostic_series"].items()},
                "shock_provenance": analysis["shock_provenance"],
                "units": {str(g): h["units"] for g, h in analysis["input_histories"].items()}}
    path = output_dir/"configuration.json"
    path.write_text(json.dumps(plain(metadata), indent=2, allow_nan=False), encoding="utf-8")
    paths["configuration"] = str(path.resolve())
    path = output_dir/"galaxy_stripping_analysis.pkl"
    with path.open("wb") as stream:
        pickle.dump(analysis, stream, protocol=pickle.HIGHEST_PROTOCOL)
    paths["analysis"] = str(path.resolve())
    if make_plots:
        import matplotlib.pyplot as plt
        plot_dir = output_dir/"diagnostics"
        plot_dir.mkdir(exist_ok=True)
        paths["galaxy_diagnostics"] = {}
        for number, gid in enumerate(analysis["diagnostic_series"]):
            figure, _ = plot_galaxy_stripping(analysis, gid)
            path = plot_dir/f"galaxy_{number:05d}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            paths["galaxy_diagnostics"][gid] = str(path.resolve())
        figure, _ = plot_stripping_population(analysis)
        path = plot_dir/"population.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths["population_diagnostics"] = str(path.resolve())
    return paths


def run_shockfinder(cell, *, minlevel=15, maxlevel=20, min_mach=1.5, show_progress=True):
    """Run ShockFinder and return the shock result plus dissipation fields."""

    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    finder.min_mach = min_mach
    finder.show_progress = show_progress

    result = finder.ShockFinder(cell)
    dissipation = pyShockFinder.compute_dissipation(cell, result)
    return result, dissipation


def shock_front_samples(result, dissipation, *, min_mach=1.5, min_flux=0.0):
    """Build positions, normals, and strengths for selected shock cells."""

    shock_mask = result.shock & (result.mach >= min_mach) & (dissipation.flux > min_flux)
    shock_rows = np.nonzero(shock_mask)[0]

    shock_pos = result.pos[shock_rows]
    shock_dx = result.dx[shock_rows]
    shock_mach = result.mach[shock_rows]
    shock_flux = dissipation.flux[shock_rows]
    shock_zone_width = (
        np.asarray(result.zone_width, dtype=np.float64)[shock_rows]
        if getattr(result, "zone_width", None) is not None
        else np.zeros(shock_rows.size, dtype=np.float64)
    )

    upstream = result.upstream_index[shock_rows]
    downstream = result.downstream_index[shock_rows]
    valid = (upstream >= 0) & (downstream >= 0)

    normal = np.zeros_like(shock_pos)
    normal[valid] = (result.normal[shock_rows[valid]] if result.normal is not None
                     else result.pos[downstream[valid]] - result.pos[upstream[valid]])
    norm = np.linalg.norm(normal, axis=1)
    valid &= norm > 0.0
    normal[valid] /= norm[valid, None]

    return {
        "rows": shock_rows,
        "pos": shock_pos,
        "dx": shock_dx,
        "mach": shock_mach,
        "flux": shock_flux,
        "zone_width": shock_zone_width,
        "normal": normal,
        "valid_normal": valid,
    }


def filter_large_shock_fronts(
    shock_catalog,
    *,
    link_length_km=50.0 * KPC_IN_KM,
    min_cells=30,
    min_extent_km=300.0 * KPC_IN_KM,
    min_total_flux=0.0,
):
    """Keep spatially extended shock components.

    This removes compact galaxy-scale SN/AGN shock islands before matching
    galaxies to shocks. Tune ``min_extent_km`` to the smallest merger-shock
    length scale you want to keep.
    """

    shock_pos = np.asarray(shock_catalog["pos"], dtype=np.float64)
    if shock_pos.size == 0:
        out = {key: value.copy() for key, value in shock_catalog.items()}
        out["component_id"] = np.empty(0, dtype=np.int64)
        out["component_size"] = np.empty(0, dtype=np.int64)
        out["component_extent"] = np.empty(0, dtype=np.float64)
        out["component_total_flux"] = np.empty(0, dtype=np.float64)
        return out

    labels = _connected_components(shock_pos, link_length_km)
    n_components = int(labels.max()) + 1

    component_size = np.bincount(labels, minlength=n_components).astype(np.int64)
    component_total_flux = np.bincount(
        labels,
        weights=np.asarray(shock_catalog["flux"], dtype=np.float64),
        minlength=n_components,
    )
    component_extent = np.zeros(n_components, dtype=np.float64)
    for component in range(n_components):
        pos = shock_pos[labels == component]
        component_extent[component] = _component_extent(pos)

    keep_component = (
        (component_size >= min_cells)
        & (component_extent >= min_extent_km)
        & (component_total_flux >= min_total_flux)
    )
    keep = keep_component[labels]

    out = {}
    for key, value in shock_catalog.items():
        arr = np.asarray(value)
        out[key] = arr[keep]
    out["component_id"] = labels[keep]
    out["component_size"] = component_size[labels[keep]]
    out["component_extent"] = component_extent[labels[keep]]
    out["component_total_flux"] = component_total_flux[labels[keep]]
    return out


def classify_galaxy_shock_crossing(
    galaxy_pos_prev,
    galaxy_pos_now,
    shock_catalog,
    *,
    search_radius_km=100.0 * KPC_IN_KM,
    width_factor=2.0,
    zone_width_factor=0.5,
    memory_budget_bytes=32 * 1024**2,
):
    """Classify whether each galaxy crossed or entered a nearby shock zone.

    Parameters
    ----------
    galaxy_pos_prev, galaxy_pos_now:
        Galaxy positions at two snapshots, in km, with matching row order.
    shock_catalog:
        Output from ``shock_front_samples``.
    search_radius_km:
        Maximum distance from the galaxy trajectory samples to a shock cell.
    width_factor:
        Geometric tolerance in units of the local shock-cell ``dx``.
    zone_width_factor:
        Multiplier applied to ``shock_catalog["zone_width"]`` when building the
        finite shock-zone half-width. The default 0.5 treats ``zone_width`` as
        the upstream-to-downstream full span around the shock center.
    """

    galaxy_pos_prev = np.asarray(galaxy_pos_prev, dtype=np.float64)
    galaxy_pos_now = np.asarray(galaxy_pos_now, dtype=np.float64)
    if galaxy_pos_now.ndim != 2 or galaxy_pos_prev.shape != galaxy_pos_now.shape or galaxy_pos_now.shape[1] != 3:
        raise ValueError("galaxy positions must both have shape (ngal, 3)")

    shock_pos = shock_catalog["pos"]
    shock_normal = shock_catalog["normal"]
    valid_normal = shock_catalog["valid_normal"]
    if shock_pos.size == 0:
        return _empty_classification(galaxy_pos_now.shape[0])

    nearest, distance = nearest_segment_shock(
        galaxy_pos_prev, galaxy_pos_now, shock_catalog,
        search_radius=search_radius_km, width_factor=width_factor,
        zone_width_factor=zone_width_factor, memory_budget_bytes=memory_budget_bytes,
    )
    catalog_zone_width = np.asarray(shock_catalog.get("zone_width", np.zeros(shock_pos.shape[0])), dtype=np.float64)
    search_zone_half_width = (
        zone_width_factor * catalog_zone_width[nearest]
        + width_factor * shock_catalog["dx"][nearest]
    )
    near = ((distance <= search_radius_km) | (distance <= search_zone_half_width)) & valid_normal[nearest]

    crossed = np.zeros(galaxy_pos_now.shape[0], dtype=bool)
    affected_zone = np.zeros(galaxy_pos_now.shape[0], dtype=bool)
    signed_prev = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    signed_now = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    zone_distance = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    zone_half_width = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    transverse = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)

    if np.any(near):
        shock_idx = nearest[near]
        p0 = galaxy_pos_prev[near]
        p1 = galaxy_pos_now[near]
        xs = shock_pos[shock_idx]
        ns = shock_normal[shock_idx]

        signed_prev[near] = np.sum((p0 - xs) * ns, axis=1)
        signed_now[near] = np.sum((p1 - xs) * ns, axis=1)

        delta_signed = signed_now[near] - signed_prev[near]
        step = p1 - p0
        length2 = np.sum(step * step, axis=1)
        fraction = np.divide(np.sum((xs - p0) * step, axis=1), length2,
                             out=np.zeros(len(p0)), where=length2 > 0)
        fraction = np.divide(-signed_prev[near], delta_signed,
                             out=fraction, where=delta_signed != 0)
        intersection = p0 + np.clip(fraction, 0, 1)[:, None] * step
        offset = intersection - xs
        normal_offset = np.sum(offset * ns, axis=1)[:, None] * ns
        transverse[near] = np.linalg.norm(offset - normal_offset, axis=1)

        transverse_width = width_factor * shock_catalog["dx"][shock_idx]
        zone_half_width[near] = zone_width_factor * catalog_zone_width[shock_idx] + transverse_width
        zone_distance[near] = _minimum_segment_plane_distance(
            signed_prev[near],
            signed_now[near],
        )
        crossed[near] = (signed_prev[near] * signed_now[near] <= 0.0) & (transverse[near] <= transverse_width)
        affected_zone[near] = (zone_distance[near] <= zone_half_width[near]) & (transverse[near] <= transverse_width)

    return {
        "crossed": crossed,
        "affected_zone": affected_zone,
        "near_shock": near,
        "nearest_shock_row": np.where(near, shock_catalog["rows"][nearest], -1),
        "nearest_component_id": _catalog_lookup(shock_catalog, "component_id", nearest, near, -1),
        "nearest_component_size": _catalog_lookup(shock_catalog, "component_size", nearest, near, -1),
        "nearest_component_extent": _catalog_lookup(shock_catalog, "component_extent", nearest, near, np.nan),
        "nearest_mach": np.where(near, shock_catalog["mach"][nearest], np.nan),
        "nearest_flux": np.where(near, shock_catalog["flux"][nearest], np.nan),
        "nearest_zone_width": np.where(near, catalog_zone_width[nearest], np.nan),
        "distance_to_shock": np.where(near, distance, np.nan),
        "signed_distance_prev": signed_prev,
        "signed_distance_now": signed_now,
        "zone_distance": zone_distance,
        "zone_half_width": zone_half_width,
        "transverse_distance": transverse,
    }


def compact_classification_results(classification, *, keep="near_or_crossed", keep_keys=None):
    """Copy only useful galaxy-classification rows into a compact result dict.

    ``keep="near_or_crossed"`` keeps galaxies flagged by ``near_shock``,
    ``crossed``, or ``affected_zone`` and records their original row numbers as
    ``galaxy_index``.
    Use ``keep="crossed"`` for the smallest post-analysis table.
    """

    if keep_keys is None:
        keep_keys = tuple(classification)

    n_galaxies = _classification_length(classification)
    keep_mask = _classification_keep_mask(classification, keep, n_galaxies)
    compact = {
        "galaxy_index": np.nonzero(keep_mask)[0].astype(np.int64, copy=False),
        "n_galaxies": np.array(n_galaxies, dtype=np.int64),
    }

    for key in keep_keys:
        value = classification[key]
        arr = np.asarray(value)
        if arr.shape[:1] == (n_galaxies,):
            compact[key] = arr[keep_mask].copy()
        else:
            compact[key] = arr.copy()
    return compact


def finalize_shock_classification(
    classification,
    *,
    result=None,
    dissipation=None,
    shock_catalog=None,
    keep="near_or_crossed",
    keep_keys=None,
):
    """Return compact classification results and release large shock arrays."""

    compact = compact_classification_results(classification, keep=keep, keep_keys=keep_keys)
    release_shock_work_arrays(result, dissipation, shock_catalog, classification)
    return compact


def release_shock_work_arrays(*objects):
    """Release arrays from temporary shock-finding objects and dictionaries."""

    for obj in objects:
        if obj is None:
            continue
        clear = getattr(obj, "clear", None)
        if callable(clear):
            clear()
        elif isinstance(obj, dict):
            obj.clear()
    gc.collect()


def _connected_components(points, link_length):
    return connected_components(points, link_length)


def _connected_components_numpy(points, link_length, chunk_size=2048):
    return connected_components(points, link_length, use_scipy=False)


def _label_neighbors(neighbors):
    labels = np.full(len(neighbors), -1, dtype=np.int64)
    label = 0
    for seed in range(len(neighbors)):
        if labels[seed] >= 0:
            continue
        labels[seed] = label
        stack = [seed]
        while stack:
            node = stack.pop()
            for other in neighbors[node]:
                if labels[other] < 0:
                    labels[other] = label
                    stack.append(other)
        label += 1
    return labels


def _component_extent(points):
    if points.shape[0] <= 1:
        return 0.0
    centered = points - np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return 0.0
    projected = centered @ vh[0]
    return float(np.max(projected) - np.min(projected))


def _catalog_lookup(shock_catalog, key, nearest, near, fill_value):
    out = np.full(near.shape[0], fill_value)
    if key in shock_catalog:
        out[near] = np.asarray(shock_catalog[key])[nearest[near]]
    return out


def _classification_length(classification):
    for key in ("near_shock", "crossed"):
        if key in classification:
            return np.asarray(classification[key]).shape[0]
    for value in classification.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            return arr.shape[0]
    raise ValueError("classification does not contain any array-like results")


def _classification_keep_mask(classification, keep, n_galaxies):
    if keep is None or keep == "all":
        return np.ones(n_galaxies, dtype=bool)
    if keep == "near_or_crossed":
        near = np.asarray(classification.get("near_shock", False), dtype=bool)
        crossed = np.asarray(classification.get("crossed", False), dtype=bool)
        affected = np.asarray(classification.get("affected_zone", False), dtype=bool)
        return near | crossed | affected
    if keep == "affected_zone":
        return np.asarray(classification["affected_zone"], dtype=bool)
    if keep == "near_shock":
        return np.asarray(classification["near_shock"], dtype=bool)
    if keep == "crossed":
        return np.asarray(classification["crossed"], dtype=bool)

    keep_mask = np.asarray(keep, dtype=bool)
    if keep_mask.shape != (n_galaxies,):
        raise ValueError("custom keep mask must have shape (n_galaxies,)")
    return keep_mask


def _nearest_shock(points, shock_pos):
    """Return nearest shock index and distance for each point."""

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nearest_shock_numpy(points, shock_pos)

    distance, nearest = cKDTree(shock_pos).query(points, workers=-1)
    return nearest.astype(np.int64), distance


def _nearest_shock_trajectory(points_prev, points_now, shock_pos):
    """Return nearest shock to the previous, current, or midpoint position."""

    points_mid = 0.5 * (points_prev + points_now)
    nearest_prev, distance_prev = _nearest_shock(points_prev, shock_pos)
    nearest_now, distance_now = _nearest_shock(points_now, shock_pos)
    nearest_mid, distance_mid = _nearest_shock(points_mid, shock_pos)

    nearest = nearest_now.copy()
    distance = distance_now.copy()

    use_prev = distance_prev < distance
    nearest[use_prev] = nearest_prev[use_prev]
    distance[use_prev] = distance_prev[use_prev]

    use_mid = distance_mid < distance
    nearest[use_mid] = nearest_mid[use_mid]
    distance[use_mid] = distance_mid[use_mid]
    return nearest, distance


def _minimum_segment_plane_distance(signed_prev, signed_now):
    crosses_plane = signed_prev * signed_now <= 0.0
    distance = np.minimum(np.abs(signed_prev), np.abs(signed_now))
    distance[crosses_plane] = 0.0
    return distance


def _nearest_shock_numpy(points, shock_pos, chunk_size=4096):
    return nearest_points(points, shock_pos)


def _empty_classification(n_galaxies):
    return {
        "crossed": np.zeros(n_galaxies, dtype=bool),
        "affected_zone": np.zeros(n_galaxies, dtype=bool),
        "near_shock": np.zeros(n_galaxies, dtype=bool),
        "nearest_shock_row": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_id": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_size": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_extent": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_mach": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_flux": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_zone_width": np.full(n_galaxies, np.nan, dtype=np.float64),
        "distance_to_shock": np.full(n_galaxies, np.nan, dtype=np.float64),
        "signed_distance_prev": np.full(n_galaxies, np.nan, dtype=np.float64),
        "signed_distance_now": np.full(n_galaxies, np.nan, dtype=np.float64),
        "zone_distance": np.full(n_galaxies, np.nan, dtype=np.float64),
        "zone_half_width": np.full(n_galaxies, np.nan, dtype=np.float64),
        "transverse_distance": np.full(n_galaxies, np.nan, dtype=np.float64),
    }


if __name__ == "__main__":
    # Replace these with your simulation data.
    cell = ...
    galaxy_pos_prev = ...   # The galaxy positions at the previous snapshot, shape (ngal, 3) in km.
    galaxy_pos_now = ...    # The galaxy positions at the current snapshot, shape (ngal, 3) in km.

    result, dissipation = run_shockfinder(cell)
    catalog = shock_front_samples(result, dissipation, min_mach=1.5)
    catalog = filter_large_shock_fronts(
        catalog,
        link_length_km=50.0 * KPC_IN_KM,
        min_cells=30,
        min_extent_km=300.0 * KPC_IN_KM,
    )
    classification = classify_galaxy_shock_crossing(
        galaxy_pos_prev,
        galaxy_pos_now,
        catalog,
        search_radius_km=100.0 * KPC_IN_KM,
    )

    print("N galaxies near shocks:", np.count_nonzero(classification["near_shock"]))
    print("N galaxies crossed shocks:", np.count_nonzero(classification["crossed"]))
    print("N galaxies affected by shock zones:", np.count_nonzero(classification["affected_zone"]))

    classification = finalize_shock_classification(
        classification,
        result=result,
        dissipation=dissipation,
        shock_catalog=catalog,
        keep="near_or_crossed",
    )
