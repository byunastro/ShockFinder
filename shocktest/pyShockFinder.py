from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    plane="xy",
    bins: int | tuple[int, int] = 1024,
    extent: tuple[float, float, float, float] | None = None,
    z_center: float | None = None,
    z_width: float | None = None,
    min_mach: float = 1.0,
    statistic="max",
    show_progress: bool = True,
    progress_interval: int = 0,
    gamma: float = 5.0 / 3.0,
    mu: float = 0.59,
):
    """Convenience wrapper returning both ``machmap`` and ``disspEmap``."""

    from shocktest import painter

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
    if extent is None:
        extent = painter.map_extent_from_result(result, plane=plane)

    machmap = painter.make_mach_map(
        result,
        plane=plane,
        bins=bins,
        extent=extent,
        z_center=z_center,
        z_width=z_width,
        min_mach=min_mach,
        statistic=statistic,
    )
    disspEmap = painter.make_disspE_map(
        result,
        dissipation,
        plane=plane,
        bins=bins,
        extent=extent,
        z_center=z_center,
        z_width=z_width,
        min_mach=min_mach,
        statistic=statistic,
    )
    return ShockMapResult(
        machmap=machmap,
        disspEmap=disspEmap,
        extent=extent,
        result=result,
        dissipation=dissipation,
    )
