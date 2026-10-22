"""Notebook-friendly functions for saved galaxy histories and shock fronts.

Load the small history tables in a cell, prepare a lazy shock search, then
classify. Large result and dissipation arrays are memory mapped only when a
selected output is requested; front member cells are discarded afterward.
"""

from __future__ import annotations

import gc
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.galaxy import (
    StrippingOptions, classify_galaxy, filter_shock_front_catalog,
    match_front_tracks, save_galaxy_stripping_analysis,
    select_nearby_front_ids, shock_product_from_catalog, shock_search_schedule,
)
from examples.saved_shock_pickle import open_saved_shock_snapshot


DEFAULT_UNITS = {"position": "kpc", "radius": "kpc", "time": "Gyr",
                 "velocity": "km/s", "coordinate_frame": "physical"}


def discover_saved_shock_snapshots(directory, start=None, end=None):
    """Find complete saved catalog/labels/result/dissipation quartets."""
    directory = Path(directory)
    if (start is None) != (end is None):
        raise ValueError("provide both snapshot range endpoints")
    if start is not None and (start < 0 or end < start):
        raise ValueError("snapshot range must satisfy 0 <= start <= end")
    snapshots = sorted(int(path.stem.split("_")[-1])
                       for path in directory.glob("catalog_[0-9][0-9][0-9][0-9][0-9].pkl"))
    if start is not None:
        snapshots = [snapshot for snapshot in snapshots if start <= snapshot <= end]
    if not snapshots:
        raise ValueError("no saved shock catalogs occur in the requested snapshot range")
    missing = [str(directory/f"{prefix}_{snapshot:05d}.pkl")
               for snapshot in snapshots for prefix in ("labels", "result", "dissipation")
               if not (directory/f"{prefix}_{snapshot:05d}.pkl").is_file()]
    if missing:
        raise FileNotFoundError(f"catalog outputs have incomplete saved products: {missing}")
    return tuple(snapshots)


def _select_fronts(catalog, positions, radii, *, snapshot, padding_kpc,
                   forced_ids, min_front_mach, min_front_cells,
                   min_front_flux, min_front_diss_rate):
    eligible = filter_shock_front_catalog(
        catalog, min_mach=min_front_mach, min_cells=min_front_cells,
        min_flux=min_front_flux, min_diss_rate=min_front_diss_rate)
    nearby = select_nearby_front_ids(eligible, positions, radii, padding_kpc=padding_kpc)
    requested = np.array(sorted(forced_ids), dtype=int)
    if len(requested):
        unknown = requested[~np.isin(requested, catalog["front_id"])]
        if len(unknown):
            raise ValueError(f"front IDs {unknown.tolist()} are absent from catalog at output {snapshot}")
        excluded = requested[~np.isin(requested, eligible["front_id"])]
        if len(excluded):
            raise ValueError(f"front IDs {excluded.tolist()} at output {snapshot} fail the front-property cuts")
    return np.union1d(nearby, requested), len(eligible), set(map(int, nearby))


@dataclass
class SavedShockSearch:
    """Selected front metadata and a callable, one-output-at-a-time loader."""

    directory: Path
    options: dict
    selected_ids: dict
    tracks: dict
    cluster_times: dict
    front_table: list
    summary: dict
    padding_kpc: float
    front_cuts: dict

    def load_product(self, snapshot):
        """Load selected member cells for one output, or None if unsearched."""
        snapshot = int(snapshot)
        if snapshot not in self.selected_ids:
            return None
        front_ids = self.selected_ids[snapshot]
        directory = self.directory
        if len(front_ids):
            with (directory/f"catalog_{snapshot:05d}.pkl").open("rb") as stream:
                catalog = pickle.load(stream)
            result, dissipation, labels = open_saved_shock_snapshot(
                directory, snapshot, position_unit="km")
            product = shock_product_from_catalog(
                catalog, labels, result, dissipation, snapshot,
                front_ids=front_ids, track_origins=self.tracks[snapshot], complete=False)
            del catalog, result, dissipation, labels
            gc.collect()
        else:
            product = {"members": [], "complete": False,
                       "provenance": {"snapshot": snapshot, "front_ids": [],
                                      "shock_origin": "unattributed"}}
        product["time_gyr"] = self.cluster_times[snapshot]
        product["provenance"].update(
            catalog_path=str(directory/f"catalog_{snapshot:05d}.pkl"),
            labels_path=str(directory/f"labels_{snapshot:05d}.pkl"),
            result_path=str(directory/f"result_{snapshot:05d}.pkl"),
            dissipation_path=str(directory/f"dissipation_{snapshot:05d}.pkl"),
            extraction_padding_kpc=self.padding_kpc,
            front_property_cuts=self.front_cuts,
            geometry_tracks=len(self.tracks[snapshot]))
        return product


def prepare_saved_shock_search(histories, cluster_info, directory, *,
                               snapshot_range=None, snapshots=None, fronts=(),
                               min_front_mach=None, min_front_cells=None,
                               min_front_flux=None, min_front_diss_rate=None,
                               padding_kpc=20., options=None):
    """Plan a shock search from already-loaded histories without reading huge arrays.

    ``fronts`` contains (snapshot, snapshot-local front ID) pairs. A requested
    front is searched for every galaxy at that output, even if it has no gas
    loss. It must still pass the property cuts. The returned ``load_product``
    method is passed directly to ``classify_galaxy``.
    """
    directory = Path(directory)
    if snapshots is not None and snapshot_range is not None:
        raise ValueError("use either snapshots or snapshot_range")
    if snapshot_range is not None:
        if len(snapshot_range) != 2:
            raise ValueError("snapshot_range must contain inclusive start and end")
        catalog_snapshots = discover_saved_shock_snapshots(directory, *snapshot_range)
    elif snapshots is None:
        catalog_snapshots = discover_saved_shock_snapshots(directory)
    else:
        catalog_snapshots = tuple(int(snapshot) for snapshot in snapshots)
        if not catalog_snapshots or tuple(sorted(set(catalog_snapshots))) != catalog_snapshots:
            raise ValueError("snapshots must be a nonempty increasing sequence of unique outputs")
        missing = [str(directory/f"{prefix}_{snapshot:05d}.pkl")
                   for snapshot in catalog_snapshots
                   for prefix in ("catalog", "labels", "result", "dissipation")
                   if not (directory/f"{prefix}_{snapshot:05d}.pkl").is_file()]
        if missing:
            raise FileNotFoundError(f"requested snapshots have incomplete saved products: {missing}")

    history_outputs = set(map(int, np.unique(np.asarray(histories["iout"]))))
    cluster_outputs = set(map(int, np.unique(np.asarray(cluster_info["iout"]))))
    no_galaxies = [s for s in catalog_snapshots if s not in history_outputs]
    no_cluster = [s for s in catalog_snapshots if s not in cluster_outputs]
    analyzable = tuple(s for s in catalog_snapshots if s in history_outputs and s in cluster_outputs)
    if not analyzable:
        raise ValueError("no saved shock output in the requested range matches galaxy and cluster histories")

    forced = {}
    for snapshot, front_id in fronts:
        if (isinstance(snapshot, (bool, np.bool_)) or isinstance(front_id, (bool, np.bool_))
                or int(snapshot) != snapshot or int(front_id) != front_id):
            raise ValueError("fronts must contain integer (snapshot, front_id) pairs")
        snapshot, front_id = int(snapshot), int(front_id)
        if snapshot not in analyzable or front_id < 0:
            raise ValueError("forced fronts require an analyzable snapshot and nonnegative ID")
        forced.setdefault(snapshot, set()).add(front_id)

    cfg = asdict(options) if isinstance(options, StrippingOptions) else dict(options or {})
    cfg.setdefault("pressure_scale", "log10")
    cfg.setdefault("shock_search_mode", "events")
    cfg["extra_shock_snapshots"] = tuple(sorted(set(cfg.get("extra_shock_snapshots", ())) | set(forced)))
    search = shock_search_schedule(histories, options=cfg)
    active_by_snapshot = {}
    for galaxy_id, outputs in search.items():
        for snapshot in outputs:
            active_by_snapshot.setdefault(snapshot, set()).add(galaxy_id)
    mapped = tuple(s for s in analyzable if s in active_by_snapshot)

    catalog_counts, selected_rows, selected, eligible_counts, nearby_ids = {}, {}, {}, {}, {}
    history_iout = np.asarray(histories["iout"])
    history_last = np.asarray(histories["last"])
    history_pos = np.asarray(histories["gal_cen"])
    dtype_names = getattr(getattr(histories, "dtype", None), "names", None)
    columns = histories.keys() if isinstance(histories, dict) else dtype_names or histories.columns
    radius_field = "gas_radius" if "gas_radius" in columns else "r90"
    history_radius = np.asarray(histories[radius_field])
    for snapshot in mapped:
        with (directory/f"catalog_{snapshot:05d}.pkl").open("rb") as stream:
            catalog = pickle.load(stream)
        rows = (history_iout == snapshot) & np.isin(history_last, tuple(active_by_snapshot[snapshot]))
        catalog_counts[snapshot] = len(catalog)
        selected[snapshot], eligible_counts[snapshot], nearby_ids[snapshot] = _select_fronts(
            catalog, history_pos[rows], history_radius[rows], snapshot=snapshot,
            padding_kpc=padding_kpc, forced_ids=forced.get(snapshot, set()),
            min_front_mach=min_front_mach, min_front_cells=min_front_cells,
            min_front_flux=min_front_flux, min_front_diss_rate=min_front_diss_rate)
        selected_rows[snapshot] = catalog[np.isin(catalog["front_id"], selected[snapshot])].copy()
        del catalog

    tracks = {snapshot: {} for snapshot in mapped}
    for previous, current in zip(mapped[:-1], mapped[1:]):
        matched = match_front_tracks(
            selected_rows[previous], selected_rows[current], selected[previous], selected[current],
            previous_tracks=tracks[previous], track_prefix=str(previous))
        tracks[current].update(matched)
        for track in matched.values():
            if track.startswith(f"{previous}:"):
                tracks[previous][int(track.split(":")[-1])] = track

    front_table = []
    for snapshot in mapped:
        for front in selected_rows[snapshot]:
            area, rate = float(front["area"]), float(front["diss_rate"])
            flux = rate/area if np.isfinite(rate) and np.isfinite(area) and area > 0 else np.nan
            front_table.append({
                "snapshot": snapshot, "front_id": int(front["front_id"]),
                "ncell": int(front["ncell"]), "mean_mach": float(front["mach"]),
                "mean_flux_erg_s_kpc2": flux, "diss_rate_erg_s": rate,
                "area_kpc2": area, "quality_bits": int(front["quality"]),
                "selected_by_spatial_filter": int(front["front_id"]) in nearby_ids[snapshot],
                "forced": int(front["front_id"]) in forced.get(snapshot, set()),
            })
    cluster_times = {int(s): float(t) for s, t in zip(cluster_info["iout"], cluster_info["t_BB"])}
    cuts = {"min_mach": min_front_mach, "min_cells": min_front_cells,
            "min_flux_erg_s_kpc2": min_front_flux, "min_diss_rate_erg_s": min_front_diss_rate}
    summary = {
        "galaxies": len(search), "mapped_snapshots": list(mapped),
        "shock_search_mode": cfg["shock_search_mode"],
        "skipped_outputs_without_candidate_loss": sorted(set(analyzable)-set(mapped)),
        "catalog_outputs_in_range": list(catalog_snapshots),
        "skipped_catalog_outputs_without_galaxies": no_galaxies,
        "skipped_catalog_outputs_without_cluster_center": no_cluster,
        "requested_snapshot_range": list(snapshot_range) if snapshot_range is not None else None,
        "outputs_without_saved_catalog_count": (
            snapshot_range[1]-snapshot_range[0]+1-len(catalog_snapshots)
            if snapshot_range is not None else None),
        "front_property_cuts": cuts,
        "catalog_fronts": {str(s): catalog_counts[s] for s in mapped},
        "eligible_fronts": {str(s): eligible_counts[s] for s in mapped},
        "selected_fronts": {str(s): len(selected[s]) for s in mapped},
        "forced_fronts": {str(s): sorted(forced.get(s, ())) for s in mapped},
        "tracked_fronts": {str(s): len(tracks[s]) for s in mapped},
        "coverage": "selected saved outputs/fronts only; non-exposure unresolved elsewhere",
        "shock_origin": "unattributed",
    }
    return SavedShockSearch(directory, cfg, selected, tracks, cluster_times,
                            front_table, summary, float(padding_kpc), cuts)


def analyze_saved_galaxies(histories, cluster_info, directory, *,
                           units=None, output_dir=None, make_plots=False, **search_kwargs):
    """Run the entire saved-front workflow from an IPython cell.

    Returns ``(analysis, search)``. Pass one branch or the combined history
    table. With ``output_dir=None`` nothing is written; plots can be displayed
    using ``plot_galaxy_stripping`` and ``plot_stripping_population``.
    """
    search = prepare_saved_shock_search(histories, cluster_info, directory, **search_kwargs)
    analysis = classify_galaxy(histories, search.load_product, cluster_info,
                               units=dict(DEFAULT_UNITS if units is None else units),
                               options=search.options, make_plots=False)
    analysis["saved_shock_selection"] = search.summary
    analysis["selected_fronts"] = search.front_table
    if output_dir is not None:
        output_dir = Path(output_dir)
        analysis["paths"] = save_galaxy_stripping_analysis(
            analysis, output_dir, make_plots=make_plots)
        front_path = output_dir/"selected_fronts.csv"
        with front_path.open("w", newline="", encoding="utf-8") as stream:
            keys = list(search.front_table[0]) if search.front_table else (
                "snapshot", "front_id", "ncell", "mean_mach", "mean_flux_erg_s_kpc2",
                "diss_rate_erg_s", "area_kpc2", "quality_bits",
                "selected_by_spatial_filter", "forced")
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(search.front_table)
        summary_path = output_dir/"saved_shock_search.json"
        summary_path.write_text(json.dumps(search.summary, indent=2), encoding="utf-8")
        analysis["paths"].update(selected_fronts=str(front_path.resolve()),
                                 saved_shock_search=str(summary_path.resolve()))
    return analysis, search
