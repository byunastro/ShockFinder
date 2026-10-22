"""Export galaxy candidates for one snapshot-local shock-front catalog ID.

Example: python -m examples.query_front_galaxies \
    artifacts/galaxy_stripping_620_900 785 3163
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from examples.galaxy import find_front_affected_galaxies, plot_galaxy_stripping


def _plain(value):
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    return value


def _write_rows(path, rows, *, fieldnames=None):
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_plain(value)) if isinstance(value, (list, tuple, np.ndarray))
                             else _plain(value) for key, value in row.items()})


def run(analysis_dir, snapshot, front_id, *, window_gyr=None,
        require_confirmed_exposure=False, plot_count=6):
    if not isinstance(plot_count, int) or plot_count < 0:
        raise ValueError("plot_count must be a nonnegative integer")
    analysis_dir = Path(analysis_dir)
    with (analysis_dir/"galaxy_stripping_analysis.pkl").open("rb") as stream:
        analysis = pickle.load(stream)
    all_rows = find_front_affected_galaxies(
        analysis, snapshot, front_id, window_gyr=window_gyr,
        include_without_loss=True,
        require_confirmed_exposure=require_confirmed_exposure)
    loss_rows = [row for row in all_rows if row["significant_loss_near_front"]]
    stem = f"front_{int(snapshot):05d}_{int(front_id)}"
    overlap_path = analysis_dir/f"{stem}_overlaps.csv"
    candidates_path = analysis_dir/f"{stem}_loss_candidates.csv"
    _write_rows(overlap_path, all_rows)
    _write_rows(candidates_path, loss_rows,
                fieldnames=list(all_rows[0]) if all_rows else None)
    plotted = (loss_rows if loss_rows else all_rows)[:plot_count]
    if plotted:
        import matplotlib.pyplot as plt

        plot_dir = analysis_dir/"diagnostics"/stem
        plot_dir.mkdir(parents=True, exist_ok=True)
        for row in plotted:
            figure, axes = plot_galaxy_stripping(analysis, row["galaxy_id"])
            for axis in axes:
                axis.axvline(row["front_time_gyr"], color="purple", lw=1.6, ls="--")
            axes[4].scatter([row["front_time_gyr"]], [row["surface_distance_kpc"]],
                            s=100, marker="*", color="purple", zorder=6,
                            label=f"selected front {snapshot}:{front_id}")
            axes[4].legend(fontsize=8)
            if np.isfinite(row["mach"]):
                axes[5].scatter([row["front_time_gyr"]], [row["mach"]],
                                s=80, marker="*", color="purple", zorder=6)
            if np.isfinite(row["dissipation_flux_erg_s_kpc2"]):
                axes[6].scatter([row["front_time_gyr"]], [row["dissipation_flux_erg_s_kpc2"]],
                                s=80, marker="*", color="purple", zorder=6)
            figure.savefig(plot_dir/f"galaxy_{row['galaxy_id']}.png", dpi=140)
            plt.close(figure)
    summary = {"snapshot": int(snapshot), "front_id": int(front_id),
               "zone_overlap_or_modeled_exposure_count": len(all_rows),
               "temporally_associated_significant_loss_count": len(loss_rows),
               "confirmed_exposure_count": sum(row["confirmed_exposure"] for row in all_rows),
               "modeled_surface_crossing_count": sum(row["modeled_surface_crossing"] for row in all_rows),
               "diagnostic_plot_count": len(plotted),
               "overlaps_csv": str(overlap_path.resolve()),
               "loss_candidates_csv": str(candidates_path.resolve()),
               "interpretation": "geometric and temporal candidates; no causal confirmation"}
    (analysis_dir/f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("snapshot", type=int)
    parser.add_argument("front_id", type=int)
    parser.add_argument("--window-gyr", type=float)
    parser.add_argument("--require-confirmed-exposure", action="store_true")
    parser.add_argument("--plot-count", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(run(args.analysis_dir, args.snapshot, args.front_id,
                         window_gyr=args.window_gyr,
                         require_confirmed_exposure=args.require_confirmed_exposure,
                         plot_count=args.plot_count), indent=2))
