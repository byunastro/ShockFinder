"""Example: dissipation-weighted Mach maps and Mach distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shocktest import painter


def dissipation_weighted_mach_distribution(
    result,
    dissipation,
    *,
    mach_bins=None,
    min_mach: float = 1.5,
    volume: float | None = None,
):
    """Return dE_diss/dlogM, optionally divided by a simulation volume."""

    if mach_bins is None:
        mach_bins = np.logspace(np.log10(min_mach), 2.5, 60)
    mach_bins = np.asarray(mach_bins, dtype=np.float64)

    shock = result.shock & (result.mach >= min_mach) & (dissipation.total > 0.0)
    mach = result.mach[shock]
    energy = dissipation.total[shock]
    hist, edges = np.histogram(np.log10(mach), bins=np.log10(mach_bins), weights=energy)
    distribution = hist / np.diff(np.log10(mach_bins))
    if volume is not None:
        distribution = distribution / float(volume)
    centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    return centers, distribution


def plot_mach_distribution(series, *, output=None):
    """Plot Figure-2-style Mach distributions for one or more runs.

    ``series`` maps labels to ``(result, dissipation, volume)`` tuples. Use
    ``volume=None`` if you do not want volume normalization.
    """

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    for label, (result, dissipation, volume) in series.items():
        mach, dist = dissipation_weighted_mach_distribution(result, dissipation, volume=volume)
        ok = dist > 0.0
        ax.plot(mach[ok], dist[ok], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{M}$")
    ax.set_ylabel(r"$dE_{\rm diss}/d\log\mathcal{M}$")
    ax.legend()
    if output is not None:
        fig.savefig(output, dpi=200)
    return fig, ax


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
# fig, ax = plot_mach_distribution({"run": (result, diss, None)})
# plt.show()
