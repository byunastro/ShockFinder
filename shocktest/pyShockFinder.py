from __future__ import annotations

import gc
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
    area: np.ndarray
    efficiency: np.ndarray
    sound_speed: np.ndarray

    def clear(self) -> None:
        """Release arrays held by this dissipation result."""

        empty = np.empty(0, dtype=np.float64)
        self.flux = empty
        self.total = empty.copy()
        self.area = empty.copy()
        self.efficiency = empty.copy()
        self.sound_speed = empty.copy()
        gc.collect()


@dataclass(slots=True)
class ShockMapResult:
    """Regular 2D maps made from AMR shock cells."""

    machmap: np.ndarray
    disspEmap: np.ndarray
    extent: tuple[float, float, float, float]
    result: shocktest.ShockResult | None
    dissipation: DissipationResult | None

    def clear(self) -> None:
        """Release map arrays and nested shock/dissipation results."""

        empty = np.empty((0, 0), dtype=np.float64)
        self.machmap = empty
        self.disspEmap = empty.copy()
        self.extent = (0.0, 0.0, 0.0, 0.0)
        if self.result is not None:
            self.result.clear()
            self.result = None
        if self.dissipation is not None:
            self.dissipation.clear()
            self.dissipation = None
        gc.collect()


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
    area_mode: str = "normal",
):
    """Compute E_diss/A in erg s^-1 kpc^-2 and cell total in erg s^-1.

    ``area_mode="normal"`` estimates the shock surface area from the local
    upstream-to-downstream normal. ``area_mode="cell"`` keeps the older ``dx^2``
    area approximation.
    """

    n = result.mach.size
    flux = np.zeros(n, dtype=np.float64)
    total = np.zeros(n, dtype=np.float64)
    area = np.zeros(n, dtype=np.float64)
    efficiency = np.zeros(n, dtype=np.float64)
    sound_speed = np.zeros(n, dtype=np.float64)

    valid = result.shock & (result.mach > 1.0) & (result.upstream_index >= 0)
    if not np.any(valid):
        return DissipationResult(flux=flux, total=total, area=area, efficiency=efficiency, sound_speed=sound_speed)

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
    area_valid = shock_surface_area(result, valid, dx_kpc, mode=area_mode)
    area[valid] = area_valid
    total[valid] = flux_valid * area_valid
    efficiency[valid] = delta
    sound_speed[valid] = cs_cgs / 1.0e5
    return DissipationResult(flux=flux, total=total, area=area, efficiency=efficiency, sound_speed=sound_speed)


def shock_surface_area(result, valid, dx_kpc, *, mode: str = "normal"):
    """Return shock surface area in kpc^2 for valid shock cells."""

    base_area = dx_kpc**2
    if mode == "cell":
        return base_area
    if mode != "normal":
        raise ValueError("area_mode must be one of: normal, cell")

    rows = np.nonzero(valid)[0]
    upstream = result.upstream_index[rows]
    downstream = result.downstream_index[rows]
    ok = (upstream >= 0) & (downstream >= 0)

    area = base_area.copy()
    if result.pos is None or not np.any(ok):
        return area

    normal = result.pos[downstream[ok]] - result.pos[upstream[ok]]
    norm = np.linalg.norm(normal, axis=1)
    ok_norm = norm > 0.0
    if not np.any(ok_norm):
        return area

    normal = normal[ok_norm] / norm[ok_norm, None]
    dominant_cos = np.max(np.abs(normal), axis=1)
    dominant_cos = np.clip(dominant_cos, 1.0 / np.sqrt(3.0), 1.0)
    ok_indices = np.nonzero(ok)[0][ok_norm]
    area[ok_indices] = base_area[ok_indices] / dominant_cos
    return area


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
    method="amr",
    show_progress: bool = True,
    progress_interval: int = 0,
    gamma: float = 5.0 / 3.0,
    mu: float = 0.59,
    area_mode: str = "normal",
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
        dissipation = compute_dissipation(cell, result, gamma=gamma, mu=mu, area_mode=area_mode)
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
        method=method,
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
        method=method,
    )
    return ShockMapResult(
        machmap=machmap,
        disspEmap=disspEmap,
        extent=extent,
        result=result,
        dissipation=dissipation,
    )
