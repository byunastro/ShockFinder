from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

import shocktest


K_BOLTZMANN = 1.380649e-16  # erg K^-1
PROTON_MASS = 1.67262192369e-24  # g
MSUN = 1.98847e33  # g
KPC = 3.0856775814913673e21  # cm

Plane = Literal["xy", "xz", "yz"]
Statistic = Literal["max", "sum", "mean"]


@dataclass(slots=True)
class DissipationResult:
    """Shock dissipation quantities, one row per retained AMR cell."""

    flux: np.ndarray
    total: np.ndarray
    efficiency: np.ndarray
    sound_speed: np.ndarray


@dataclass(slots=True)
class ShockMapResult:
    """Regular 2D maps made from AMR shock cells."""

    machmap: np.ndarray
    disspEmap: np.ndarray
    extent: tuple[float, float, float, float]
    result: shocktest.ShockResult
    dissipation: DissipationResult


def compute_shock_result(
    cell,
    *,
    minlevel: int = 13,
    maxlevel: int = 20,
    show_progress: bool = True,
    progress_interval: int = 0,
):
    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    finder.show_progress = show_progress
    finder.progress_interval = progress_interval
    return finder.ShockFinder(cell)


def thermalization_efficiency(mach, gamma: float = 5.0 / 3.0):
    """Rankine-Hugoniot thermalization efficiency delta(M)."""

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


def compute_dissipation(
    cell,
    result,
    *,
    gamma: float = 5.0 / 3.0,
    mu: float = 0.59,
):
    """Compute E_diss/A in erg s^-1 kpc^-2 and cell total in erg s^-1."""

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

    # 0.5 rho v^3 is erg s^-1 cm^-2. Multiplying by kpc^2 gives
    # the plotting unit used in many shock papers: erg s^-1 kpc^-2.
    flux_valid = 0.5 * rho_cgs * (mach * cs_cgs) ** 3 * delta * KPC**2

    flux[valid] = flux_valid
    dx_kpc = dx / (KPC / 1.0e5)
    total[valid] = flux_valid * dx_kpc**2
    efficiency[valid] = delta
    sound_speed[valid] = cs_cgs / 1.0e5
    return DissipationResult(flux=flux, total=total, efficiency=efficiency, sound_speed=sound_speed)


def make_shock_maps(
    cell,
    *,
    result=None,
    dissipation: DissipationResult | None = None,
    minlevel: int = 13,
    maxlevel: int = 20,
    plane: Plane = "xy",
    bins: int | tuple[int, int] = 1024,
    extent: tuple[float, float, float, float] | None = None,
    z_center: float | None = None,
    z_width: float | None = None,
    min_mach: float = 1.0,
    statistic: Statistic = "max",
    show_progress: bool = True,
    progress_interval: int = 0,
    gamma: float = 5.0 / 3.0,
    mu: float = 0.59,
):
    """Return ``machmap`` and ``disspEmap`` regular images from AMR shocks.

    ``disspEmap`` is E_diss/A in erg s^-1 kpc^-2. Empty pixels are NaN.
    """

    if result is None:
        result = compute_shock_result(
            cell,
            minlevel=minlevel,
            maxlevel=maxlevel,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
    if dissipation is None:
        dissipation = compute_dissipation(cell, result, gamma=gamma, mu=mu)

    x_key, y_key, normal_key = _plane_keys(plane)
    rows = result.selected_indices
    x = np.asarray(cell[x_key], dtype=np.float64)[rows]
    y = np.asarray(cell[y_key], dtype=np.float64)[rows]
    normal = np.asarray(cell[normal_key], dtype=np.float64)[rows]
    dx = np.asarray(cell["dx", "km"], dtype=np.float64)[rows]

    if extent is None:
        extent = (
            float(np.min(x - 0.5 * dx)),
            float(np.max(x + 0.5 * dx)),
            float(np.min(y - 0.5 * dx)),
            float(np.max(y + 0.5 * dx)),
        )

    draw = result.shock & (result.mach >= min_mach) & (dissipation.flux > 0.0)
    if z_center is not None and z_width is not None:
        draw &= np.abs(normal - z_center) <= 0.5 * z_width

    machmap = _bin_to_map(x[draw], y[draw], result.mach[draw], bins, extent, statistic)
    disspEmap = _bin_to_map(x[draw], y[draw], dissipation.flux[draw], bins, extent, statistic)
    return ShockMapResult(
        machmap=machmap,
        disspEmap=disspEmap,
        extent=extent,
        result=result,
        dissipation=dissipation,
    )


def _plane_keys(plane: Plane):
    axes = {
        "xy": (("x", "km"), ("y", "km"), ("z", "km")),
        "xz": (("x", "km"), ("z", "km"), ("y", "km")),
        "yz": (("y", "km"), ("z", "km"), ("x", "km")),
    }
    if plane not in axes:
        raise ValueError("plane must be one of: xy, xz, yz")
    return axes[plane]


def _bin_shape(bins: int | tuple[int, int]):
    if isinstance(bins, int):
        return bins, bins
    if len(bins) != 2:
        raise ValueError("bins must be an int or (ny, nx)")
    return int(bins[0]), int(bins[1])


def _bin_to_map(x, y, values, bins, extent, statistic: Statistic):
    ny, nx = _bin_shape(bins)
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    if values.size == 0:
        return out

    xmin, xmax, ymin, ymax = extent
    ix = np.floor((x - xmin) / (xmax - xmin) * nx).astype(np.int64)
    iy = np.floor((y - ymin) / (ymax - ymin) * ny).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(values)
    if not np.any(inside):
        return out

    flat = iy[inside] * nx + ix[inside]
    vals = values[inside]
    if statistic == "max":
        work = np.full(ny * nx, -np.inf, dtype=np.float64)
        np.maximum.at(work, flat, vals)
        work[~np.isfinite(work)] = np.nan
        return work.reshape(ny, nx)
    if statistic == "sum":
        work = np.zeros(ny * nx, dtype=np.float64)
        np.add.at(work, flat, vals)
        work[work == 0.0] = np.nan
        return work.reshape(ny, nx)
    if statistic == "mean":
        work = np.zeros(ny * nx, dtype=np.float64)
        count = np.zeros(ny * nx, dtype=np.int64)
        np.add.at(work, flat, vals)
        np.add.at(count, flat, 1)
        valid = count > 0
        work[valid] /= count[valid]
        work[~valid] = np.nan
        return work.reshape(ny, nx)
    raise ValueError("statistic must be one of: max, sum, mean")
