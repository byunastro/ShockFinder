"""Plot and summarize ShockFinder Mach cross-validation diagnostics.

Call ``plot_mach_validation(result)`` after running a finder with its default
``validate_mach=True``.  The function returns aggregate reason counts and AMR
level pass rates in addition to creating the four-panel diagnostic figure.
"""

from __future__ import annotations

import numpy as np


def summarize_mach_validation(result) -> dict[str, object]:
    """Return consistency counts and per-level pass fractions."""

    required = (
        "pressure_check_valid",
        "pressure_consistent",
        "density_check_applicable",
        "density_consistent",
        "mach_consistent",
    )
    if any(getattr(result, name, None) is None for name in required):
        raise ValueError("result has no Mach validation arrays")
    shock = np.asarray(result.mach, dtype=np.float64) > 1.0
    pressure_valid = np.asarray(result.pressure_check_valid, dtype=bool)
    pressure_pass = np.asarray(result.pressure_consistent, dtype=bool)
    density_applicable = np.asarray(result.density_check_applicable, dtype=bool)
    density_pass = np.asarray(result.density_consistent, dtype=bool)
    overall = np.asarray(result.mach_consistent, dtype=bool)

    reason_counts = {
        "pressure_pass": int(np.count_nonzero(shock & pressure_pass)),
        "pressure_fail": int(
            np.count_nonzero(shock & pressure_valid & ~pressure_pass)
        ),
        "pressure_unavailable": int(
            np.count_nonzero(shock & ~pressure_valid)
        ),
        "density_pass": int(np.count_nonzero(shock & density_pass)),
        "density_fail": int(
            np.count_nonzero(shock & density_applicable & ~density_pass)
        ),
        "density_not_applicable": int(
            np.count_nonzero(shock & ~density_applicable)
        ),
        "overall_pass": int(np.count_nonzero(shock & overall)),
        "overall_fail": int(np.count_nonzero(shock & ~overall)),
    }
    level_rates: dict[int, dict[str, float | int]] = {}
    if getattr(result, "level", None) is not None:
        level = np.asarray(result.level)
        for value in np.unique(level[shock]):
            selected = shock & (level == value)
            count = int(np.count_nonzero(selected))
            level_rates[int(value)] = {
                "count": count,
                "pass_fraction": float(np.count_nonzero(selected & overall) / count),
            }
    return {"reason_counts": reason_counts, "level_rates": level_rates}


def plot_mach_validation(
    result,
    *,
    consistency_factor: float = 1.5,
    mach_bins=None,
    output=None,
):
    """Create MT--MP, MT--Mrho, MP/MT, and binned-pass plots."""

    import matplotlib.pyplot as plt

    summary = summarize_mach_validation(result)
    mt = np.asarray(result.mach_temperature, dtype=np.float64)
    mp = np.asarray(result.mach_pressure, dtype=np.float64)
    md = np.asarray(result.mach_density, dtype=np.float64)
    shock = mt > 1.0
    density_applicable = np.asarray(result.density_check_applicable, dtype=bool)
    overall = np.asarray(result.mach_consistent, dtype=bool)
    finite_p = shock & np.isfinite(mp)
    finite_d = shock & np.isfinite(md)
    if mach_bins is None:
        mach_bins = np.logspace(0.0, 2.0, 13)
    mach_bins = np.asarray(mach_bins, dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    if np.any(finite_p):
        limits = np.array([
            min(np.min(mt[finite_p]), np.min(mp[finite_p])),
            max(np.max(mt[finite_p]), np.max(mp[finite_p])),
        ])
        axes[0, 0].scatter(mt[finite_p], mp[finite_p], s=5, alpha=0.3)
        axes[0, 0].plot(limits, limits, color="black", label=r"$M_T=M_P$")
        axes[0, 0].plot(limits, limits * consistency_factor, "k--", alpha=0.6)
        axes[0, 0].plot(limits, limits / consistency_factor, "k--", alpha=0.6)
    axes[0, 0].set(xscale="log", yscale="log", xlabel=r"$M_T$", ylabel=r"$M_P$")
    axes[0, 0].legend()

    applicable = finite_d & density_applicable
    not_applicable = finite_d & ~density_applicable
    axes[0, 1].scatter(mt[applicable], md[applicable], s=6, alpha=0.4, label="checked")
    axes[0, 1].scatter(mt[not_applicable], md[not_applicable], s=6, alpha=0.25,
                       label="not applicable")
    if np.any(finite_d):
        limits = np.array([
            min(np.min(mt[finite_d]), np.min(md[finite_d])),
            max(np.max(mt[finite_d]), np.max(md[finite_d])),
        ])
        axes[0, 1].plot(limits, limits, color="black", label=r"$M_T=M_\rho$")
        axes[0, 1].plot(limits, limits * consistency_factor, "k--", alpha=0.6)
        axes[0, 1].plot(limits, limits / consistency_factor, "k--", alpha=0.6)
    axes[0, 1].set(xscale="log", yscale="log", xlabel=r"$M_T$", ylabel=r"$M_\rho$")
    axes[0, 1].legend()

    axes[1, 0].hist(mp[finite_p] / mt[finite_p], bins=60, histtype="step")
    axes[1, 0].axvspan(
        1.0 / consistency_factor, consistency_factor, color="tab:green", alpha=0.15
    )
    axes[1, 0].set(xlabel=r"$M_P/M_T$", ylabel="shock count")

    centers = np.sqrt(mach_bins[:-1] * mach_bins[1:])
    rates = np.full(centers.size, np.nan)
    for index, (lower, upper) in enumerate(zip(mach_bins[:-1], mach_bins[1:])):
        selected = shock & (mt >= lower) & (mt < upper)
        if np.any(selected):
            rates[index] = np.mean(overall[selected])
    axes[1, 1].plot(centers, rates, marker="o")
    axes[1, 1].set(xscale="log", ylim=(0.0, 1.05), xlabel=r"$M_T$",
                   ylabel="consistency pass fraction")

    if output is not None:
        fig.savefig(output, dpi=180)
    return fig, summary


__all__ = ["plot_mach_validation", "summarize_mach_validation"]
