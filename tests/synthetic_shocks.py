from __future__ import annotations

# Based on Ryu et al. 2003, ApJ, 593, 599.

from dataclasses import dataclass

import numpy as np


GAMMA = 5.0 / 3.0


@dataclass(frozen=True, slots=True)
class ShockSpec:
    mach: float
    index: int
    upstream_temperature: float = 1.0e4
    upstream_density: float = 1.0e6
    level: int = 20


def temperature_jump_from_mach(mach: float) -> float:
    """Rankine-Hugoniot T2/T1 for gamma=5/3."""

    m2 = float(mach) ** 2
    return ((5.0 * m2 - 1.0) * (m2 + 3.0)) / (16.0 * m2)


def density_jump_from_mach(mach: float) -> float:
    """Rankine-Hugoniot rho2/rho1 for gamma=5/3."""

    m2 = float(mach) ** 2
    return (4.0 * m2) / (m2 + 3.0)


def mach_from_temperature_jump(t_ratio: np.ndarray | float) -> np.ndarray:
    """Invert the temperature-jump Mach relation."""

    ratio = np.asarray(t_ratio, dtype=np.float64)
    disc = (14.0 - 16.0 * ratio) ** 2 + 60.0
    m2 = ((16.0 * ratio - 14.0) + np.sqrt(disc)) / 10.0
    return np.sqrt(np.maximum(m2, 1.0))


def planar_shock_cell(
    mach: float,
    *,
    n: int = 12,
    shock_index: int | None = None,
    upstream_temperature: float = 1.0e4,
    upstream_density: float = 1.0e6,
    level: int = 20,
):
    """Return a 1D planar shock embedded in AMR-cell table fields."""

    if shock_index is None:
        shock_index = n // 2
    if shock_index < 2 or shock_index > n - 3:
        raise ValueError("shock_index must leave at least two cells on each side")

    x = np.arange(n, dtype=np.float64) + 0.5
    temp = np.full(n, upstream_temperature, dtype=np.float64)
    rho = np.full(n, upstream_density, dtype=np.float64)

    post = slice(shock_index, None)
    temp[post] = upstream_temperature * temperature_jump_from_mach(mach)
    rho[post] = upstream_density * density_jump_from_mach(mach)

    vx = np.full(n, 100.0, dtype=np.float64)
    vx[post] = -100.0

    return cell_from_1d_profiles(x, temp, rho, vx, level=level)


def multi_shock_line_cell(
    *,
    n: int = 96,
    specs: tuple[ShockSpec, ...] = (
        ShockSpec(mach=8.0, index=16, upstream_temperature=1.0e4, upstream_density=0.5e6),
        ShockSpec(mach=3.0, index=36, upstream_temperature=2.0e6, upstream_density=5.0e6),
        ShockSpec(mach=2.5, index=58, upstream_temperature=3.0e6, upstream_density=6.0e6),
        ShockSpec(mach=6.0, index=78, upstream_temperature=1.0e4, upstream_density=0.7e6),
    ),
):
    """Line with external and internal shocks."""

    x = np.arange(n, dtype=np.float64) + 0.5
    temp = np.full(n, 1.0e4, dtype=np.float64)
    rho = np.full(n, 0.5e6, dtype=np.float64)
    vx = np.full(n, 80.0, dtype=np.float64)

    for spec in specs:
        idx = spec.index
        temp[idx - 2 : idx] = spec.upstream_temperature
        rho[idx - 2 : idx] = spec.upstream_density
        temp[idx:] = max(temp[idx], spec.upstream_temperature * temperature_jump_from_mach(spec.mach))
        rho[idx:] = max(rho[idx], spec.upstream_density * density_jump_from_mach(spec.mach))
        vx[idx - 2 : idx] = 80.0
        vx[idx:] = -80.0

        if idx + 3 < n:
            vx[idx + 3 :] = 80.0
            temp[idx + 3 :] = max(temp[idx + 3], spec.upstream_temperature * 1.15)
            rho[idx + 3 :] = max(rho[idx + 3], spec.upstream_density * 1.15)

    return cell_from_1d_profiles(x, temp, rho, vx)


def cell_from_1d_profiles(x, temp, rho, vx, *, level: int = 20):
    n = np.asarray(x).size
    return {
        ("x", "km"): np.asarray(x, dtype=np.float64),
        ("y", "km"): np.full(n, 0.5, dtype=np.float64),
        ("z", "km"): np.full(n, 0.5, dtype=np.float64),
        ("dx", "km"): np.ones(n, dtype=np.float64),
        ("vx", "km/s"): np.asarray(vx, dtype=np.float64),
        ("vy", "km/s"): np.zeros(n, dtype=np.float64),
        ("vz", "km/s"): np.zeros(n, dtype=np.float64),
        ("T", "K"): np.asarray(temp, dtype=np.float64),
        ("rho", "Msol/kpc3"): np.asarray(rho, dtype=np.float64),
        "level": np.full(n, level, dtype=np.int32),
    }


def tiled_sheet_cell(base_cell, *, ny: int = 16, nz: int = 1):
    """Tile a 1D shock line into a 2D/3D sheet so maps and areas are meaningful."""

    x1 = np.asarray(base_cell["x", "km"], dtype=np.float64)
    n1 = x1.size
    total = n1 * ny * nz
    x = np.empty(total, dtype=np.float64)
    y = np.empty(total, dtype=np.float64)
    z = np.empty(total, dtype=np.float64)

    cursor = 0
    for iz in range(nz):
        for iy in range(ny):
            block = slice(cursor, cursor + n1)
            x[block] = x1
            y[block] = iy + 0.5
            z[block] = iz + 0.5
            cursor += n1

    out = {
        ("x", "km"): x,
        ("y", "km"): y,
        ("z", "km"): z,
        ("dx", "km"): np.ones(total, dtype=np.float64),
        "level": np.full(total, int(base_cell["level"][0]), dtype=np.int32),
    }
    for key in [("vx", "km/s"), ("vy", "km/s"), ("vz", "km/s"), ("T", "K"), ("rho", "Msol/kpc3")]:
        out[key] = np.tile(np.asarray(base_cell[key], dtype=np.float64), ny * nz)
    return out


def shock_surface_histogram(result, *, bins: np.ndarray, min_mach: float = 1.5, area_factor: float = 1.19):
    """Return a dS/dlogM proxy from detected shock centers."""

    mach = result.mach[result.shock & (result.mach >= min_mach)]
    if mach.size == 0:
        return np.zeros(bins.size - 1, dtype=np.float64)
    dx = result.dx[result.shock & (result.mach >= min_mach)]
    weights = area_factor * dx**2
    hist, _ = np.histogram(np.log10(mach), bins=np.log10(bins), weights=weights)
    dlog = np.diff(np.log10(bins))
    return hist / dlog


def external_internal_masks(cell, result, *, threshold: float = 1.0e4, min_mach: float = 1.5):
    valid = result.shock & (result.mach >= min_mach) & (result.upstream_index >= 0)
    retained = result.selected_indices
    upstream_rows = retained[result.upstream_index]
    upstream_temp = np.asarray(cell["T", "K"], dtype=np.float64)[upstream_rows]
    external = valid & (upstream_temp <= threshold)
    internal = valid & (upstream_temp > threshold)
    return external, internal
