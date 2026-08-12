from __future__ import annotations

from collections import Counter

import numpy as np

from .catalog import ShockCatalog


def summarize_catalog_quality(catalog: ShockCatalog) -> dict[str, object]:
    """Return compact completeness, classification, and quality diagnostics."""

    groups = catalog.groups
    flags = Counter(flag for group in groups for flag in group.quality_flags)
    classifications = Counter(group.classification for group in groups)
    return {
        "n_groups": len(groups),
        "n_complete": sum(group.is_complete for group in groups),
        "n_boundary": sum(group.touches_boundary for group in groups),
        "surface_area": float(sum(group.area for group in groups)),
        "dissipation_total": float(sum(group.dissipation_total for group in groups)),
        "classifications": dict(sorted(classifications.items())),
        "quality_flags": dict(sorted(flags.items())),
    }


def plot_catalog_quality(catalog: ShockCatalog, *, figsize=(12, 8)):
    """Plot group Mach, size, normal dispersion, and completeness diagnostics."""

    import matplotlib.pyplot as plt

    groups = catalog.groups
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if not groups:
        for axis in axes.ravel():
            axis.text(0.5, 0.5, "No shock groups", ha="center", va="center")
            axis.set_axis_off()
        return fig, axes

    mach = np.asarray([group.mach_peak for group in groups])
    area = np.asarray([group.area for group in groups])
    dispersion = np.asarray([group.normal_dispersion for group in groups])
    centers = np.asarray([group.n_centers for group in groups])
    complete = np.asarray([group.is_complete for group in groups])

    axes[0, 0].hist(mach, bins="auto")
    axes[0, 0].set(xlabel="Peak Mach", ylabel="Groups")
    axes[0, 1].scatter(mach, area, c=np.where(complete, "C0", "C3"), s=18)
    axes[0, 1].set(xlabel="Peak Mach", ylabel="Surface area")
    axes[1, 0].scatter(centers, dispersion, c=np.where(complete, "C0", "C3"), s=18)
    axes[1, 0].set(xlabel="Center count", ylabel="Normal dispersion")
    labels = ["complete", "incomplete"]
    values = [int(np.count_nonzero(complete)), int(np.count_nonzero(~complete))]
    axes[1, 1].bar(labels, values, color=["C0", "C3"])
    axes[1, 1].set(ylabel="Groups")
    return fig, axes
