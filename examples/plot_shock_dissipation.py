from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

import shocktest


K_BOLTZMANN = 1.380649e-16  # erg K^-1
PROTON_MASS = 1.67262192369e-24  # g
MSUN = 1.98847e33  # g
KPC = 3.0856775814913673e21  # cm


@dataclass(slots=True)
class DissipationResult:
    """Shock dissipation quantities, one row per retained AMR cell."""

    flux: np.ndarray
    total: np.ndarray
    efficiency: np.ndarray
    sound_speed: np.ndarray


def thermalization_efficiency(mach, gamma: float = 5.0 / 3.0):
    """Rankine-Hugoniot thermalization efficiency delta(M).

    This is the ideal-gas efficiency used in
    E_diss/A = 0.5 * rho_1 * (M c_s,1)^3 * delta(M).
    """

    mach = np.asarray(mach, dtype=np.float64)
    efficiency = np.zeros_like(mach)
    mask = mach > 1.0
    m2 = mach[mask] ** 2
    compression = ((gamma + 1.0) * m2) / ((gamma - 1.0) * m2 + 2.0)
    pressure_jump = (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)
    efficiency[mask] = (
        2.0
        / (gamma * (gamma - 1.0) * m2 * compression)
        * (pressure_jump - compression**gamma)
    )
    efficiency[efficiency < 0.0] = 0.0
    return efficiency


def compute_shock_result(cell, minlevel: int = 13, maxlevel: int = 20, show_progress: bool = True):
    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    finder.show_progress = show_progress
    return finder.ShockFinder(cell)


def compute_dissipation(
    cell,
    result,
    *,
    gamma: float = 5.0 / 3.0,
    mu: float = 0.59,
):
    """Compute shock dissipation flux and total power.

    Parameters
    ----------
    cell:
        AMR cell table with fields used by ``shocktest``.
    result:
        ``shocktest.ShockResult``.
    gamma:
        Adiabatic index.
    mu:
        Mean molecular weight. ``0.59`` is a common fully ionized primordial
        value. Change it if your temperature/sound-speed convention differs.

    Returns
    -------
    DissipationResult
        ``flux`` is E_diss/A in ``erg s^-1 kpc^-2``. ``total`` is E_diss per
        AMR shock cell in ``erg s^-1`` using area ``dx^2``.
    """

    n = result.mach.size
    flux = np.zeros(n, dtype=np.float64)
    total = np.zeros(n, dtype=np.float64)
    efficiency = np.zeros(n, dtype=np.float64)
    sound_speed = np.zeros(n, dtype=np.float64)

    valid = result.shock & (result.mach > 1.0) & (result.upstream_index >= 0)
    if not np.any(valid):
        return DissipationResult(flux=flux, total=total, efficiency=efficiency, sound_speed=sound_speed)

    retained_rows = result.selected_indices
    upstream_rows = retained_rows[result.upstream_index[valid]]
    center_rows = retained_rows[np.nonzero(valid)[0]]

    temp1 = np.asarray(cell["T", "K"], dtype=np.float64)[upstream_rows]
    rho1 = np.asarray(cell["rho", "Msol/kpc3"], dtype=np.float64)[upstream_rows]
    dx = np.asarray(cell["dx", "km"], dtype=np.float64)[center_rows]
    mach = result.mach[valid]

    rho_cgs = rho1 * MSUN / KPC**3
    cs_cgs = np.sqrt(gamma * K_BOLTZMANN * temp1 / (mu * PROTON_MASS))
    delta = thermalization_efficiency(mach, gamma=gamma)

    # 0.5 rho v^3 is an energy flux in erg s^-1 cm^-2. Multiplying by
    # kpc^2 converts it to the figure-friendly erg s^-1 kpc^-2.
    flux_valid = 0.5 * rho_cgs * (mach * cs_cgs) ** 3 * delta * KPC**2

    flux[valid] = flux_valid
    dx_kpc = dx / (KPC / 1.0e5)
    total[valid] = flux_valid * dx_kpc**2
    efficiency[valid] = delta
    sound_speed[valid] = cs_cgs / 1.0e5
    return DissipationResult(flux=flux, total=total, efficiency=efficiency, sound_speed=sound_speed)


def plot_dissipation_and_mach(
    cell,
    result,
    dissipation,
    *,
    plane: str = "xy",
    z_center: float | None = None,
    z_width: float | None = None,
    min_mach: float = 1.0,
    log_ediss_range: tuple[float, float] | None = None,
    log_mach_range: tuple[float, float] | None = None,
    circle_center: tuple[float, float] | None = None,
    circle_radius: float | None = None,
    scale_bar_length: float | None = 1000.0 * KPC / 1.0e5,
    scale_bar_label: str = "1 Mpc",
    output: str | Path | None = "shock_dissipation_mach.png",
):
    """Draw side-by-side maps of log dissipation flux and log Mach number."""

    axes = {
        "xy": (("x", "km"), ("y", "km"), ("z", "km")),
        "xz": (("x", "km"), ("z", "km"), ("y", "km")),
        "yz": (("y", "km"), ("z", "km"), ("x", "km")),
    }
    if plane not in axes:
        raise ValueError("plane must be one of: xy, xz, yz")

    x_key, y_key, normal_key = axes[plane]
    rows = result.selected_indices
    x = np.asarray(cell[x_key], dtype=np.float64)[rows]
    y = np.asarray(cell[y_key], dtype=np.float64)[rows]
    normal = np.asarray(cell[normal_key], dtype=np.float64)[rows]
    dx = np.asarray(cell["dx", "km"], dtype=np.float64)[rows]

    draw = result.shock & (result.mach >= min_mach) & (dissipation.flux > 0.0)
    if z_center is not None and z_width is not None:
        draw &= np.abs(normal - z_center) <= 0.5 * z_width

    fig, axes_obj = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax in axes_obj:
        ax.set_aspect("equal")
        ax.set_xlabel(f"{x_key[0]} [{x_key[1]}]")
        ax.set_ylabel(f"{y_key[0]} [{y_key[1]}]")

    if not np.any(draw):
        for ax in axes_obj:
            ax.text(0.5, 0.5, "No shock cells in selected slice", transform=ax.transAxes, ha="center")
        if output is not None:
            fig.savefig(output, dpi=200)
        return fig, axes_obj

    patches = [
        Rectangle((x_i - 0.5 * dx_i, y_i - 0.5 * dx_i), dx_i, dx_i)
        for x_i, y_i, dx_i in zip(x[draw], y[draw], dx[draw])
    ]
    log_ediss = np.log10(dissipation.flux[draw])
    log_mach = np.log10(result.mach[draw])

    left = PatchCollection(patches, array=log_ediss, cmap="plasma", edgecolor="none", rasterized=True)
    right = PatchCollection(
        [Rectangle(p.get_xy(), p.get_width(), p.get_height()) for p in patches],
        array=log_mach,
        cmap="RdYlBu_r",
        edgecolor="none",
        rasterized=True,
    )
    if log_ediss_range is not None:
        left.set_clim(*log_ediss_range)
    if log_mach_range is not None:
        right.set_clim(*log_mach_range)

    axes_obj[0].add_collection(left)
    axes_obj[1].add_collection(right)
    axes_obj[0].set_title(r"Shock dissipation flux")
    axes_obj[1].set_title(r"Mach number")

    pad = 0.5 * np.max(dx[draw])
    xmin = np.min(x[draw] - 0.5 * dx[draw]) - pad
    xmax = np.max(x[draw] + 0.5 * dx[draw]) + pad
    ymin = np.min(y[draw] - 0.5 * dx[draw]) - pad
    ymax = np.max(y[draw] + 0.5 * dx[draw]) + pad
    for ax in axes_obj:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if circle_center is not None and circle_radius is not None:
            ax.add_patch(Circle(circle_center, circle_radius, fill=False, ls=":", lw=1.5, color="white"))
        if scale_bar_length is not None:
            _draw_scale_bar(ax, scale_bar_length, scale_bar_label)

    cbar0 = fig.colorbar(left, ax=axes_obj[0], orientation="horizontal", pad=0.08)
    cbar0.set_label(r"$\log E_{\rm diss}\ [{\rm erg\ s^{-1}\ kpc^{-2}}]$")
    cbar1 = fig.colorbar(right, ax=axes_obj[1], orientation="horizontal", pad=0.08)
    cbar1.set_label(r"$\log \mathcal{M}$")

    if output is not None:
        fig.savefig(output, dpi=200)
    return fig, axes_obj


def _draw_scale_bar(ax, length: float, label: str):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmax - 0.08 * (xmax - xmin) - length
    x1 = x0 + length
    y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot([x0, x1], [y0, y0], color="white", lw=4, solid_capstyle="butt")
    ax.text(0.5 * (x0 + x1), y0 + 0.025 * (ymax - ymin), label, color="white",
            ha="center", va="bottom", fontsize=12, weight="bold")


# Example use after your simulation code has created `cell`:
#
# from examples.plot_shock_dissipation import (
#     compute_shock_result,
#     compute_dissipation,
#     plot_dissipation_and_mach,
# )
#
# result = compute_shock_result(cell, minlevel=13, maxlevel=20, show_progress=True)
# diss = compute_dissipation(cell, result, mu=0.59)
# plot_dissipation_and_mach(
#     cell,
#     result,
#     diss,
#     plane="xy",
#     min_mach=1.0,
#     log_ediss_range=(37, 42),
#     log_mach_range=(0.3, 1.7),
#     scale_bar_length=1000.0 * KPC / 1.0e5,
#     scale_bar_label="1 Mpc",
#     output="shock_dissipation_mach.png",
# )
# plt.show()
