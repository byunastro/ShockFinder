"""Run galaxy/shock analysis over a chosen saved-output range and front cuts.

Only complete saved product quartets inside the requested range are loaded.
Selected fronts are a property/spatial subset, so omitted snapshots and
fronts never establish non-exposure. No front is assigned a merger origin.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
from pathlib import Path

import numpy as np

from examples.galaxy import (
    classify_galaxy, filter_shock_front_catalog, match_front_tracks, plot_galaxy_stripping,
    plot_stripping_population, select_nearby_front_ids,
    shock_product_from_catalog, shock_search_schedule,
)
from examples.saved_shock_pickle import open_saved_shock_snapshot


def discover_saved_shock_snapshots(directory, start=None, end=None):
    """List catalog outputs in an inclusive range, requiring paired products."""
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


def run(directory, output_dir, *, snapshots=None, snapshot_range=None,
        min_front_mach=None, min_front_cells=None, min_front_flux=None,
        min_front_diss_rate=None, padding_kpc=20., diagnostic_count=12,
        forced_fronts=(), shock_search_mode="events"):
    directory, output_dir = Path(directory), Path(output_dir)
    if snapshots is not None and snapshot_range is not None:
        raise ValueError("use either snapshots or snapshot_range")
    if snapshot_range is not None:
        if len(snapshot_range) != 2:
            raise ValueError("snapshot_range must contain inclusive start and end")
        snapshots = discover_saved_shock_snapshots(directory, *snapshot_range)
    elif snapshots is None:
        snapshots = discover_saved_shock_snapshots(directory)
    else:
        snapshots = tuple(int(snapshot) for snapshot in snapshots)
        if not snapshots or sorted(set(snapshots)) != list(snapshots):
            raise ValueError("snapshots must be a nonempty increasing sequence of unique outputs")
        missing = [str(directory/f"{prefix}_{snapshot:05d}.pkl")
                   for snapshot in snapshots
                   for prefix in ("catalog", "labels", "result", "dissipation")
                   if not (directory/f"{prefix}_{snapshot:05d}.pkl").is_file()]
        if missing:
            raise FileNotFoundError(f"requested snapshots have incomplete saved products: {missing}")
    catalog_snapshots = snapshots
    with (directory/"dwarf_catalog_620_900.pkl").open("rb") as stream:
        histories = pickle.load(stream)
    with (directory/"cluster_info.pkl").open("rb") as stream:
        cluster_info = pickle.load(stream)
    history_outputs = set(map(int, np.unique(histories["iout"])))
    cluster_outputs = set(map(int, np.unique(cluster_info["iout"])))
    no_galaxies = [s for s in snapshots if s not in history_outputs]
    no_cluster = [s for s in snapshots if s not in cluster_outputs]
    analyzable_snapshots = tuple(s for s in snapshots if s in history_outputs and s in cluster_outputs)
    if not analyzable_snapshots:
        raise ValueError("no saved shock output in the requested range matches galaxy and cluster histories")
    forced = {}
    for snapshot, front_id in forced_fronts:
        snapshot, front_id = int(snapshot), int(front_id)
        if snapshot not in analyzable_snapshots or front_id < 0:
            raise ValueError("forced fronts require an analyzable snapshot and nonnegative ID")
        forced.setdefault(snapshot, set()).add(front_id)
    options = {"pressure_scale": "log10", "shock_search_mode": shock_search_mode,
               "extra_shock_snapshots": tuple(forced)}
    search = shock_search_schedule(histories, options=options)
    active_by_snapshot = {}
    for galaxy_id, outputs in search.items():
        for snapshot in outputs:
            active_by_snapshot.setdefault(snapshot, set()).add(galaxy_id)
    snapshots = tuple(s for s in analyzable_snapshots if s in active_by_snapshot)
    catalog_counts, selected_rows, selected, eligible_counts, nearby_ids = {}, {}, {}, {}, {}
    for snapshot in snapshots:
        with (directory/f"catalog_{snapshot:05d}.pkl").open("rb") as stream:
            catalog = pickle.load(stream)
        rows = histories[(histories["iout"] == snapshot)
                         & np.isin(histories["last"], tuple(active_by_snapshot[snapshot]))]
        catalog_counts[snapshot] = len(catalog)
        selected[snapshot], eligible_counts[snapshot], nearby_ids[snapshot] = _select_fronts(
            catalog, rows["gal_cen"], rows["r90"], snapshot=snapshot,
            padding_kpc=padding_kpc, forced_ids=forced.get(snapshot, set()),
            min_front_mach=min_front_mach, min_front_cells=min_front_cells,
            min_front_flux=min_front_flux, min_front_diss_rate=min_front_diss_rate)
        selected_rows[snapshot] = catalog[np.isin(catalog["front_id"], selected[snapshot])].copy()
        del catalog
    tracks = {snapshot: {} for snapshot in snapshots}
    for previous, current in zip(snapshots[:-1], snapshots[1:]):
        matched = match_front_tracks(
            selected_rows[previous], selected_rows[current], selected[previous], selected[current],
            previous_tracks=tracks[previous], track_prefix=str(previous))
        tracks[current].update(matched)
        for track in matched.values():
            if track.startswith(f"{previous}:"):
                tracks[previous][int(track.split(":")[-1])] = track
    def load_product(snapshot):
        if snapshot not in selected:
            return None
        if len(selected[snapshot]):
            with (directory/f"catalog_{snapshot:05d}.pkl").open("rb") as stream:
                catalog = pickle.load(stream)
            result, dissipation, labels = open_saved_shock_snapshot(
                directory, snapshot, position_unit="km")
            product = shock_product_from_catalog(
                catalog, labels, result, dissipation, snapshot,
                front_ids=selected[snapshot], track_origins=tracks[snapshot],
                complete=False)
            del catalog, result, dissipation, labels
            gc.collect()
        else:
            product = {"members": [], "complete": False,
                       "provenance": {"snapshot": int(snapshot), "front_ids": [],
                                      "shock_origin": "unattributed"}}
        product["time_gyr"] = float(cluster_info["t_BB"][np.flatnonzero(cluster_info["iout"] == snapshot)[0]])
        product["provenance"].update(
            catalog_path=str(directory/f"catalog_{snapshot:05d}.pkl"),
            labels_path=str(directory/f"labels_{snapshot:05d}.pkl"),
            result_path=str(directory/f"result_{snapshot:05d}.pkl"),
            dissipation_path=str(directory/f"dissipation_{snapshot:05d}.pkl"),
            extraction_padding_kpc=padding_kpc,
            front_property_cuts={"min_mach": min_front_mach, "min_cells": min_front_cells,
                                 "min_flux_erg_s_kpc2": min_front_flux,
                                 "min_diss_rate_erg_s": min_front_diss_rate},
            geometry_tracks=len(tracks[snapshot]))
        return product
    units = {"position": "kpc", "radius": "kpc", "time": "Gyr",
             "velocity": "km/s", "coordinate_frame": "physical"}
    analysis = classify_galaxy(
        histories, load_product, cluster_info, units=units,
        options=options, output_dir=output_dir, make_plots=False)
    import matplotlib.pyplot as plt

    diagnostic_dir = output_dir/"diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    loss_rows = sorted(analysis["classifications"],
                       key=lambda row: row["fractional_loss"] if np.isfinite(row["fractional_loss"]) else -1.,
                       reverse=True)
    encounter_rows = sorted((row for row in analysis["shock_encounters"] if row["confirmed"]),
                            key=lambda row: row["confidence_score"], reverse=True)
    diagnostic_ids = list(dict.fromkeys(
        [row["galaxy_id"] for row in loss_rows[:max(1, diagnostic_count//2)]]
        + [row["galaxy_id"] for row in encounter_rows]
        + [row["galaxy_id"] for row in loss_rows]))[:diagnostic_count]
    for galaxy_id in diagnostic_ids:
        figure, _ = plot_galaxy_stripping(analysis, galaxy_id)
        figure.savefig(diagnostic_dir/f"galaxy_{galaxy_id}.png", dpi=140)
        plt.close(figure)
    figure, _ = plot_stripping_population(analysis)
    figure.savefig(diagnostic_dir/"population.png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"selected_fronts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("snapshot", "front_id", "ncell", "mean_mach", "mean_flux_erg_s_kpc2",
                         "diss_rate_erg_s", "area_kpc2", "quality_bits", "selected_by_spatial_filter",
                         "forced"))
        for snapshot in snapshots:
            for front in selected_rows[snapshot]:
                area, rate = float(front["area"]), float(front["diss_rate"])
                flux = rate/area if np.isfinite(rate) and np.isfinite(area) and area > 0 else np.nan
                writer.writerow((snapshot, int(front["front_id"]), int(front["ncell"]),
                                 float(front["mach"]), flux, rate, area, int(front["quality"]),
                                 int(front["front_id"]) in nearby_ids[snapshot],
                                 int(front["front_id"]) in forced.get(snapshot, set())))
    summary = {"galaxies": len(analysis["classifications"]),
               "events": len(analysis["gas_loss_events"]),
               "significant_events": sum(row["significant"] for row in analysis["gas_loss_events"]),
               "categories": analysis["population"]["category_counts"],
               "mapped_snapshots": list(snapshots),
               "shock_search_mode": shock_search_mode,
               "skipped_outputs_without_candidate_loss": sorted(set(analyzable_snapshots)-set(snapshots)),
               "catalog_outputs_in_range": list(catalog_snapshots),
               "skipped_catalog_outputs_without_galaxies": no_galaxies,
               "skipped_catalog_outputs_without_cluster_center": no_cluster,
               "requested_snapshot_range": list(snapshot_range) if snapshot_range is not None else None,
               "outputs_without_saved_catalog_count": (
                   snapshot_range[1]-snapshot_range[0]+1-len(catalog_snapshots)
                   if snapshot_range is not None else None),
               "front_property_cuts": {"min_mach": min_front_mach, "min_cells": min_front_cells,
                                       "min_flux_erg_s_kpc2": min_front_flux,
                                       "min_diss_rate_erg_s": min_front_diss_rate},
               "catalog_fronts": {str(s): catalog_counts[s] for s in snapshots},
               "eligible_fronts": {str(s): eligible_counts[s] for s in snapshots},
               "selected_fronts": {str(s): len(selected[s]) for s in snapshots},
               "selected_fronts_csv": str((output_dir/"selected_fronts.csv").resolve()),
               "forced_fronts": {str(s): sorted(forced.get(s, ())) for s in snapshots},
               "tracked_fronts": {str(s): len(tracks[s]) for s in snapshots},
               "coverage": "selected saved outputs/fronts only; non-exposure unresolved elsewhere",
               "shock_origin": "unattributed"}
    (output_dir/"run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    def front_pair(value):
        try:
            snapshot, front_id = (int(part) for part in value.split(":"))
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError("use SNAPSHOT:FRONT_ID, for example 785:3163") from None
        return snapshot, front_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--snapshot-range", nargs=2, type=int, metavar=("START", "END"),
                        help="inclusive range; use saved product outputs found within it")
    parser.add_argument("--min-front-mach", type=float,
                        help="minimum catalog mean Mach number (inclusive)")
    parser.add_argument("--min-front-cells", type=int,
                        help="minimum number of member shock cells (inclusive)")
    parser.add_argument("--min-front-flux", type=float,
                        help="minimum mean dissipation flux [erg/s/kpc2] (inclusive)")
    parser.add_argument("--min-front-diss-rate", type=float,
                        help="minimum integrated dissipation rate [erg/s] (inclusive)")
    parser.add_argument("--padding-kpc", type=float, default=20.)
    parser.add_argument("--diagnostic-count", type=int, default=12)
    parser.add_argument("--shock-search-mode", choices=("events", "all"), default="events",
                        help="search only near gas-loss episodes, or all available outputs")
    parser.add_argument("--front", type=front_pair, action="append", default=[],
                        help="force extraction of a snapshot-local front; repeatable, e.g. 785:3163")
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.output_dir,
                         snapshot_range=args.snapshot_range,
                         min_front_mach=args.min_front_mach,
                         min_front_cells=args.min_front_cells,
                         min_front_flux=args.min_front_flux,
                         min_front_diss_rate=args.min_front_diss_rate,
                         padding_kpc=args.padding_kpc,
                         diagnostic_count=args.diagnostic_count,
                         forced_fronts=args.front,
                         shock_search_mode=args.shock_search_mode), indent=2))
