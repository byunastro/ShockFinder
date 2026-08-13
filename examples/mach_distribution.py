"""Example: dissipation-weighted Mach maps and Mach distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shocktest import painter


def print_shock_summary(result, dissipation) -> None:
    """Print quick diagnostics before plotting Mach distributions."""

    shock = np.asarray(result.shock, dtype=bool)
    mach = np.asarray(result.mach, dtype=np.float64)
    flux = np.asarray(dissipation.flux, dtype=np.float64)
    total = np.asarray(dissipation.total, dtype=np.float64)

    print(f"retained cells: {mach.size}")
    print(f"selected input cells: {result.selected_indices.size}")
    print(f"shock cells: {np.count_nonzero(shock)}")
    print(f"mach > 0 cells: {np.count_nonzero(np.isfinite(mach) & (mach > 0.0))}")
    print(f"mach >= 1.5 shock cells: {np.count_nonzero(shock & np.isfinite(mach) & (mach >= 1.5))}")
    print(f"positive dissipation flux cells: {np.count_nonzero(np.isfinite(flux) & (flux > 0.0))}")
    print(f"positive dissipation total cells: {np.count_nonzero(np.isfinite(total) & (total > 0.0))}")
    if mach.size:
        print(f"max mach: {np.nanmax(mach)}")
    if flux.size:
        print(f"max dissipation flux: {np.nanmax(flux)}")
    if total.size:
        print(f"max dissipation total: {np.nanmax(total)}")


def dissipation_weighted_mach_distribution(
    result,
    dissipation,
    *,
    mach_bins=None,
    min_mach: float = 1.5,
    volume: float | None = None,
    weight_field: str = "total",
):
    """Return dE_diss/dlogM, optionally divided by a simulation volume.

    ``weight_field="total"`` uses cell-integrated dissipation. If that is zero
    for your data, try ``weight_field="flux"`` to use dissipation flux instead.
    """

    mach_all = np.asarray(result.mach, dtype=np.float64)
    energy_all = np.asarray(getattr(dissipation, weight_field), dtype=np.float64)
    shock = (
        result.shock
        & np.isfinite(mach_all)
        & (mach_all >= min_mach)
        & np.isfinite(energy_all)
        & (energy_all > 0.0)
    )
    if not np.any(shock):
        shock_count = int(np.count_nonzero(result.shock))
        mach_count = int(np.count_nonzero(result.shock & np.isfinite(mach_all) & (mach_all >= min_mach)))
        energy_count = int(np.count_nonzero(np.isfinite(energy_all) & (energy_all > 0.0)))
        raise ValueError(
            "No shocks have positive dissipation in the requested Mach range. "
            f"shock_count={shock_count}, mach_above_min_count={mach_count}, "
            f"positive_{weight_field}_count={energy_count}. "
            'Try a smaller min_mach or weight_field="flux".'
        )

    mach = mach_all[shock]
    energy = energy_all[shock]
    if mach_bins is None:
        mach_hi = float(np.nanmax(mach))
        mach_hi = max(mach_hi * 1.001, min_mach * 1.01)
        mach_bins = np.logspace(np.log10(min_mach), np.log10(mach_hi), 60)
    else:
        mach_bins = np.asarray(mach_bins, dtype=np.float64)
        if np.any(mach_bins <= 0.0) or np.any(np.diff(mach_bins) <= 0.0):
            raise ValueError("mach_bins must be positive and strictly increasing")

    hist, edges = np.histogram(np.log10(mach), bins=np.log10(mach_bins), weights=energy)
    distribution = hist / np.diff(np.log10(mach_bins))
    if volume is not None:
        distribution = distribution / float(volume)
    centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    return centers, distribution


# After running ShockFinder and compute_dissipation:
#
# weighted_machmap = painter.make_mach_map(
#     result,
#     plane="xz",
#     bins=1000,
#     statistic="mean",
#     method="amr",
#     weights=diss.total,
# )
# print_shock_summary(result, diss)
# fig, ax = plot_current_mach_distribution(result, diss)
# plt.show()
