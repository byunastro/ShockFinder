"""Compare ShockFinder parameter sweeps without retaining all results in memory.

The script reads one result/dissipation pair at a time and writes:

* ``shockfinder_summary.csv``: scalar diagnostics for every setting
* ``shockfinder_distributions.npz``: common-bin Mach distributions
* ``summary_heatmaps.png``: parameter sensitivity of key scalar diagnostics
* ``mach_distributions.png``: count-, area-, and dissipation-weighted PDFs
* ``fiducial_relative_changes.png``: changes relative to one fiducial setting
* ``cellwise_mach_heatmaps.png``: exact-cell Mach agreement with the fiducial
* ``cellwise_mach_comparisons.npz``: 2D Mach-vs-Mach histograms

Example
-------
python examples/compare_shockfinder_settings.py \
    --data-dir /storage1/byunkh/NC_map/shockmap \
    --output-dir /storage1/byunkh/NC_map/shockmap/comparison_00785

The default loader is ``rur.utool.load``.  Use ``--loader pickle`` if the files
are ordinary pickle files and ``rur`` is not available.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KPC_IN_KM = 3.0856775814913673e16
MINLEVELS = (12, 13, 14)
MAXLEVELS = (18, 19, 20)
MIN_TEMPERATURES = (1.0e4, 1.0e5, 1.0e6)


@dataclass(frozen=True, order=True)
class Setting:
    minlevel: int
    maxlevel: int
    min_temperature: float

    @property
    def temperature_exponent(self) -> int:
        return int(round(np.log10(self.min_temperature)))

    @property
    def suffix(self) -> str:
        return (
            f"lmin{self.minlevel}_lmax{self.maxlevel}"
            f"_Tmin1e{self.temperature_exponent}"
        )

    @property
    def label(self) -> str:
        return (
            rf"$l_{{\min}}={self.minlevel}$, "
            rf"$l_{{\max}}={self.maxlevel}$, "
            rf"$T_{{\min}}=10^{{{self.temperature_exponent}}}$ K"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep", type=int, default=785)
    parser.add_argument(
        "--fiducial", nargs=3, metavar=("LMIN", "LMAX", "TMIN"),
        default=(13, 19, 1.0e5), type=float,
        help="fiducial minlevel maxlevel min_temperature",
    )
    parser.add_argument(
        "--loader", default="rur.utool",
        help="module containing load(path), or 'pickle'",
    )
    parser.add_argument("--min-mach", type=float, default=1.0)
    parser.add_argument("--max-mach-bin", type=float, default=1.0e3)
    parser.add_argument("--n-mach-bins", type=int, default=60)
    return parser.parse_args()


def make_loader(name: str):
    if name == "pickle":
        def load_pickle(path: Path):
            with path.open("rb") as stream:
                return pickle.load(stream)
        return load_pickle

    module = importlib.import_module(name)
    if not hasattr(module, "load"):
        raise AttributeError(f"{name!r} has no load(path) function")
    return lambda path: module.load(str(path))


def paths_for(data_dir: Path, timestep: int, setting: Setting):
    tag = f"{timestep:05d}_{setting.suffix}"
    return (
        data_dir / f"result_{tag}.pkl",
        data_dir / f"dissipation_{tag}.pkl",
    )


def finite_sum(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.sum(selected[np.isfinite(selected)], dtype=np.float64))


def finite_quantile(values: np.ndarray, mask: np.ndarray, q: float) -> float:
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    return float(np.quantile(selected, q)) if selected.size else np.nan


def cubed_length_sum_kpc3(
    dx_km: np.ndarray, mask: np.ndarray | None = None, chunk_size: int = 5_000_000
) -> float:
    """Sum cell volumes while keeping temporary arrays below about 40 MiB."""
    total = 0.0
    for start in range(0, dx_km.size, chunk_size):
        stop = min(start + chunk_size, dx_km.size)
        chunk = dx_km[start:stop]
        if mask is not None:
            chunk = chunk[mask[start:stop]]
        scaled = chunk / KPC_IN_KM
        total += float(np.sum(scaled * scaled * scaled, dtype=np.float64))
    return total


def summarize(result, diss, setting: Setting, mach_edges: np.ndarray):
    mach = np.asarray(result.mach)
    shock = np.asarray(result.shock, dtype=bool)
    total = np.asarray(diss.total)
    flux = np.asarray(diss.flux)
    area = np.asarray(diss.area)
    dx = np.asarray(result.dx) if result.dx is not None else None

    valid = shock & np.isfinite(mach) & (mach > 0.0)
    dissipating = valid & np.isfinite(total) & (total > 0.0)
    n_retained = int(mach.size)
    n_shock = int(np.count_nonzero(valid))

    # selected_indices is sorted and unique because it comes from np.nonzero.
    shock_original_indices = np.asarray(result.selected_indices)[valid]

    count_hist, _ = np.histogram(mach[valid], bins=mach_edges)
    area_hist, _ = np.histogram(mach[valid], bins=mach_edges, weights=area[valid])
    diss_hist, _ = np.histogram(
        mach[dissipating], bins=mach_edges, weights=total[dissipating]
    )
    dlogm = np.diff(np.log10(mach_edges))

    if dx is not None:
        # ShockFinder's default position_unit is km. Convert dx^3 to kpc^3.
        retained_volume = cubed_length_sum_kpc3(dx)
        shock_volume = cubed_length_sum_kpc3(dx, valid)
    else:
        retained_volume = np.nan
        shock_volume = np.nan

    summary = {
        "minlevel": setting.minlevel,
        "maxlevel": setting.maxlevel,
        "min_temperature": setting.min_temperature,
        "n_retained": n_retained,
        "n_shock": n_shock,
        "shock_number_fraction": n_shock / n_retained if n_retained else np.nan,
        "shock_volume_kpc3": shock_volume,
        "retained_volume_kpc3": retained_volume,
        "shock_volume_fraction": (
            shock_volume / retained_volume if retained_volume > 0.0 else np.nan
        ),
        "mach_mean": finite_sum(mach, valid) / n_shock if n_shock else np.nan,
        "mach_median": finite_quantile(mach, valid, 0.5),
        "mach_p90": finite_quantile(mach, valid, 0.9),
        "mach_p99": finite_quantile(mach, valid, 0.99),
        "mach_max": float(np.nanmax(mach[valid])) if n_shock else np.nan,
        "n_mach_ge_2": int(np.count_nonzero(valid & (mach >= 2.0))),
        "n_mach_ge_5": int(np.count_nonzero(valid & (mach >= 5.0))),
        "n_mach_ge_10": int(np.count_nonzero(valid & (mach >= 10.0))),
        "total_area_kpc2": finite_sum(area, valid),
        "total_dissipation_erg_s": finite_sum(total, dissipating),
        "mean_positive_flux_erg_s_kpc2": (
            finite_sum(flux, dissipating) / np.count_nonzero(dissipating)
            if np.any(dissipating) else np.nan
        ),
        "n_positive_dissipation": int(np.count_nonzero(dissipating)),
        "mach_underflow_count": int(np.count_nonzero(valid & (mach < mach_edges[0]))),
        "mach_overflow_count": int(np.count_nonzero(valid & (mach >= mach_edges[-1]))),
    }
    distributions = {
        "count_dndlogm": count_hist / dlogm,
        "area_dadlogm": area_hist / dlogm,
        "diss_dedlogm": diss_hist / dlogm,
    }
    return summary, distributions, shock_original_indices, mach[valid].copy()


def sampled_spearman(x: np.ndarray, y: np.ndarray, max_samples: int = 1_000_000):
    """Spearman coefficient on an evenly spaced sample to bound memory/time."""
    if x.size < 2:
        return np.nan, int(x.size)
    if x.size > max_samples:
        take = np.linspace(0, x.size - 1, max_samples, dtype=np.int64)
        x = x[take]
        y = y[take]
    # Mach values are effectively continuous. Average-rank tie handling would
    # be much more expensive and has negligible impact for these arrays.
    rank_x = np.empty(x.size, dtype=np.int64)
    rank_y = np.empty(y.size, dtype=np.int64)
    rank_x[np.argsort(x, kind="mergesort")] = np.arange(x.size)
    rank_y[np.argsort(y, kind="mergesort")] = np.arange(y.size)
    return float(np.corrcoef(rank_x, rank_y)[0, 1]), int(x.size)


def compare_with_fiducial(
    indices: np.ndarray,
    mach: np.ndarray,
    fiducial_indices: np.ndarray,
    fiducial_mach: np.ndarray,
    mach_edges: np.ndarray,
):
    if np.array_equal(indices, fiducial_indices):
        common = int(indices.size)
        mach_test = mach
        mach_fid = fiducial_mach
    else:
        common_ids, test_pos, fid_pos = np.intersect1d(
            indices, fiducial_indices, assume_unique=True, return_indices=True
        )
        common = int(common_ids.size)
        del common_ids
        mach_test = mach[test_pos]
        mach_fid = fiducial_mach[fid_pos]
        del test_pos, fid_pos

    # summarize() already restricted both arrays to finite positive-Mach shocks.
    union = indices.size + fiducial_indices.size - common
    metrics = {
        "fiducial_common_shocks": common,
        "fiducial_jaccard": common / union if union else 1.0,
        "fiducial_recall": common / fiducial_indices.size if fiducial_indices.size else np.nan,
        "fiducial_precision": common / indices.size if indices.size else np.nan,
    }
    if not mach_fid.size:
        metrics.update({
            "mach_common_valid": 0,
            "mach_relative_bias_pct": np.nan,
            "mach_relative_median_pct": np.nan,
            "mach_absolute_relative_median_pct": np.nan,
            "mach_absolute_relative_p90_pct": np.nan,
            "mach_log10_rmse_dex": np.nan,
            "mach_pearson_r": np.nan,
            "mach_spearman_r_sampled": np.nan,
            "mach_spearman_sample_size": 0,
            "mach_agree_within_1pct": np.nan,
            "mach_agree_within_5pct": np.nan,
            "mach_agree_within_10pct": np.nan,
            "mach_agree_within_20pct": np.nan,
        })
        shape = (mach_edges.size - 1, mach_edges.size - 1)
        return metrics, np.zeros(shape, dtype=np.int64)

    relative = mach_test / mach_fid - 1.0
    absolute_relative = np.abs(relative)
    log_ratio = np.log10(mach_test / mach_fid)
    spearman, spearman_n = sampled_spearman(mach_fid, mach_test)
    metrics.update({
        "mach_common_valid": int(mach_fid.size),
        "mach_relative_bias_pct": float(np.mean(relative) * 100.0),
        "mach_relative_median_pct": float(np.median(relative) * 100.0),
        "mach_absolute_relative_median_pct": float(np.median(absolute_relative) * 100.0),
        "mach_absolute_relative_p90_pct": float(np.quantile(absolute_relative, 0.9) * 100.0),
        "mach_log10_rmse_dex": float(np.sqrt(np.mean(log_ratio * log_ratio))),
        "mach_pearson_r": float(np.corrcoef(mach_fid, mach_test)[0, 1]),
        "mach_spearman_r_sampled": spearman,
        "mach_spearman_sample_size": spearman_n,
        "mach_agree_within_1pct": float(np.mean(absolute_relative <= 0.01 + 1.0e-12)),
        "mach_agree_within_5pct": float(np.mean(absolute_relative <= 0.05 + 1.0e-12)),
        "mach_agree_within_10pct": float(np.mean(absolute_relative <= 0.10 + 1.0e-12)),
        "mach_agree_within_20pct": float(np.mean(absolute_relative <= 0.20 + 1.0e-12)),
    })
    del relative, absolute_relative, log_ratio

    # Accumulate in chunks so log10 conversion never duplicates all matched
    # Mach values at once. Axis 0 is fiducial Mach; axis 1 is test Mach.
    joint = np.zeros((mach_edges.size - 1, mach_edges.size - 1), dtype=np.int64)
    chunk_size = 2_000_000
    for start in range(0, mach_fid.size, chunk_size):
        stop = min(start + chunk_size, mach_fid.size)
        hist, _, _ = np.histogram2d(
            mach_fid[start:stop], mach_test[start:stop],
            bins=(mach_edges, mach_edges),
        )
        joint += hist.astype(np.int64)
    return metrics, joint


def clear_large_objects(result, diss) -> None:
    if hasattr(result, "clear"):
        result.clear()
    if hasattr(diss, "clear"):
        diss.clear()
    del result, diss
    gc.collect()


def process_one(load, data_dir, timestep, setting, mach_edges):
    result_path, diss_path = paths_for(data_dir, timestep, setting)
    if not result_path.exists() or not diss_path.exists():
        missing = [str(p) for p in (result_path, diss_path) if not p.exists()]
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    started = time.perf_counter()
    result = load(result_path)
    diss = load(diss_path)
    try:
        output = summarize(result, diss, setting, mach_edges)
    finally:
        clear_large_objects(result, diss)
    print(f"Processed {setting.suffix} in {time.perf_counter() - started:.1f} s")
    return output


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap(ax, rows, metric, temperature, title, *, log_color=False):
    grid = np.full((len(MINLEVELS), len(MAXLEVELS)), np.nan)
    for row in rows:
        if row["min_temperature"] == temperature:
            i = MINLEVELS.index(row["minlevel"])
            j = MAXLEVELS.index(row["maxlevel"])
            grid[i, j] = row[metric]
    shown = np.log10(grid) if log_color else grid
    image = ax.imshow(shown, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(MAXLEVELS)), MAXLEVELS)
    ax.set_yticks(range(len(MINLEVELS)), MINLEVELS)
    ax.set_xlabel("maxlevel")
    ax.set_ylabel("minlevel")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, label=("log10 " if log_color else "") + metric)


def plot_heatmaps(rows, output_dir: Path) -> None:
    metrics = (
        ("n_shock", True),
        ("total_dissipation_erg_s", True),
        ("shock_volume_fraction", False),
        ("fiducial_jaccard", False),
    )
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 16), constrained_layout=True)
    for col, temperature in enumerate(MIN_TEMPERATURES):
        for row_index, (metric, log_color) in enumerate(metrics):
            heatmap(
                axes[row_index, col], rows, metric, temperature,
                rf"$T_{{\min}}=10^{{{int(np.log10(temperature))}}}$ K",
                log_color=log_color,
            )
    fig.savefig(output_dir / "summary_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_distributions(settings, distributions, mach_edges, output_dir: Path) -> None:
    centers = np.sqrt(mach_edges[:-1] * mach_edges[1:])
    keys = ("count_dndlogm", "area_dadlogm", "diss_dedlogm")
    labels = (
        r"$dN/d\log_{10}M$",
        r"$dA/d\log_{10}M$ [kpc$^2$]",
        r"$dE_{\rm diss}/d\log_{10}M$ [erg s$^{-1}$]",
    )
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharex=True, constrained_layout=True)
    colors = {12: "tab:blue", 13: "tab:orange", 14: "tab:green"}
    linestyles = {18: ":", 19: "--", 20: "-"}
    for col, temperature in enumerate(MIN_TEMPERATURES):
        for setting in settings:
            if setting.min_temperature != temperature:
                continue
            for row, (key, ylabel) in enumerate(zip(keys, labels)):
                values = distributions[setting.suffix][key]
                positive = values > 0.0
                axes[row, col].plot(
                    centers[positive], values[positive],
                    color=colors[setting.minlevel],
                    linestyle=linestyles[setting.maxlevel],
                    alpha=0.85,
                    label=f"lmin={setting.minlevel}, lmax={setting.maxlevel}",
                )
                axes[row, col].set_xscale("log")
                axes[row, col].set_yscale("log")
                axes[row, col].set_ylabel(ylabel)
                axes[row, col].grid(alpha=0.2)
        axes[0, col].set_title(
            rf"$T_{{\min}}=10^{{{int(np.log10(temperature))}}}$ K"
        )
        axes[-1, col].set_xlabel("Mach number")
    axes[0, -1].legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "mach_distributions.png", dpi=180)
    plt.close(fig)


def plot_relative_changes(rows, fiducial: Setting, output_dir: Path) -> None:
    fid = next(
        row for row in rows
        if row["minlevel"] == fiducial.minlevel
        and row["maxlevel"] == fiducial.maxlevel
        and row["min_temperature"] == fiducial.min_temperature
    )
    metrics = (
        "n_shock", "shock_volume_kpc3", "total_area_kpc2",
        "total_dissipation_erg_s", "mach_median", "mach_p90",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [
        f"{r['minlevel']}/{r['maxlevel']}/1e{int(np.log10(r['min_temperature']))}"
        for r in rows
    ]
    for ax, metric in zip(axes.flat, metrics):
        base = fid[metric]
        relative = np.array([row[metric] / base - 1.0 for row in rows]) * 100.0
        ax.bar(x, relative)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(metric)
        ax.set_ylabel("change from fiducial [%]")
        ax.set_xticks(x, labels, rotation=90, fontsize=6)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "fiducial_relative_changes.png", dpi=180)
    plt.close(fig)


def plot_cellwise_mach_heatmaps(rows, output_dir: Path) -> None:
    metrics = (
        ("mach_absolute_relative_median_pct", False),
        ("mach_absolute_relative_p90_pct", False),
        ("mach_log10_rmse_dex", False),
        ("mach_agree_within_5pct", False),
        ("mach_pearson_r", False),
        ("mach_spearman_r_sampled", False),
    )
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 22), constrained_layout=True)
    for col, temperature in enumerate(MIN_TEMPERATURES):
        for row_index, (metric, log_color) in enumerate(metrics):
            heatmap(
                axes[row_index, col], rows, metric, temperature,
                rf"$T_{{\min}}=10^{{{int(np.log10(temperature))}}}$ K",
                log_color=log_color,
            )
    fig.savefig(output_dir / "cellwise_mach_heatmaps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load = make_loader(args.loader)
    fiducial = Setting(int(args.fiducial[0]), int(args.fiducial[1]), args.fiducial[2])
    settings = [
        Setting(lmin, lmax, temperature)
        for temperature in MIN_TEMPERATURES
        for lmin in MINLEVELS
        for lmax in MAXLEVELS
    ]
    if fiducial not in settings:
        raise ValueError(f"Fiducial setting {fiducial} is not in the parameter grid")
    mach_edges = np.logspace(
        np.log10(args.min_mach), np.log10(args.max_mach_bin), args.n_mach_bins + 1
    )

    # Load the fiducial first. Its compact original-row index vector is enough
    # to compute exact spatial overlap with every later setting.
    ordered = [fiducial] + [setting for setting in settings if setting != fiducial]
    rows = []
    distributions = {}
    cellwise_histograms = {}
    fiducial_indices = None
    fiducial_mach = None
    for setting in ordered:
        summary, distribution, shock_indices, shock_mach = process_one(
            load, args.data_dir, args.timestep, setting, mach_edges
        )
        if setting == fiducial:
            fiducial_indices = shock_indices.copy()
            fiducial_mach = shock_mach.copy()
        comparison, joint_histogram = compare_with_fiducial(
            shock_indices, shock_mach, fiducial_indices, fiducial_mach, mach_edges
        )
        summary.update(comparison)
        rows.append(summary)
        distributions[setting.suffix] = distribution
        cellwise_histograms[setting.suffix] = joint_histogram
        del shock_indices, shock_mach
        gc.collect()

    # Restore a stable grid order in tables and figures.
    rank = {setting: index for index, setting in enumerate(settings)}
    rows.sort(key=lambda row: rank[Setting(
        row["minlevel"], row["maxlevel"], row["min_temperature"]
    )])
    write_summary(args.output_dir / "shockfinder_summary.csv", rows)
    np.savez_compressed(
        args.output_dir / "shockfinder_distributions.npz",
        mach_edges=mach_edges,
        **{
            f"{suffix}__{key}": values
            for suffix, dist in distributions.items()
            for key, values in dist.items()
        },
    )
    np.savez_compressed(
        args.output_dir / "cellwise_mach_comparisons.npz",
        mach_edges=mach_edges,
        **{
            f"{suffix}__joint_count": histogram
            for suffix, histogram in cellwise_histograms.items()
        },
    )
    plot_heatmaps(rows, args.output_dir)
    plot_distributions(settings, distributions, mach_edges, args.output_dir)
    plot_relative_changes(rows, fiducial, args.output_dir)
    plot_cellwise_mach_heatmaps(rows, args.output_dir)
    print(f"Wrote comparison products to {args.output_dir}")


if __name__ == "__main__":
    main()
