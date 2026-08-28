"""Rankine--Hugoniot Mach estimators and cross-validation helpers.

The temperature jump remains ShockFinder's primary Mach estimator.  Pressure
and density jumps are independent diagnostics for ideal, adiabatic,
hydrodynamic shocks.  In particular, the density jump approaches the finite
strong-shock compression ``(gamma + 1) / (gamma - 1)`` and is consequently a
poor high-Mach estimator.
"""

from __future__ import annotations

from enum import IntFlag

import numpy as np


class MachValidationFlag(IntFlag):
    """Bit meanings stored in ``ShockResult.mach_validation_status``."""

    PRESSURE_VALID = 1 << 0
    PRESSURE_CONSISTENT = 1 << 1
    PRESSURE_FROM_RHO_T = 1 << 2
    DENSITY_VALID = 1 << 3
    DENSITY_APPLICABLE = 1 << 4
    DENSITY_CONSISTENT = 1 << 5
    DENSITY_SATURATED = 1 << 6
    MACH_CONSISTENT = 1 << 7
    ENDPOINT_INVALID = 1 << 8


def _as_float_array(value):
    return np.asarray(value, dtype=np.float64)


def jump_ratio(upstream, downstream):
    """Return ``downstream / upstream`` or NaN for invalid endpoint values."""

    upstream, downstream = np.broadcast_arrays(
        _as_float_array(upstream), _as_float_array(downstream)
    )
    ratio = np.full(upstream.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(upstream)
        & np.isfinite(downstream)
        & (upstream > 0.0)
        & (downstream > 0.0)
    )
    np.divide(downstream, upstream, out=ratio, where=valid)
    return ratio


def mach_from_temperature_ratio(ratio, gamma: float = 5.0 / 3.0):
    """Analytically invert the ideal-gas temperature jump.

    Unlike the compiled detection kernel, which maps a non-shock ratio to
    Mach 1 before applying its candidate cuts, this public diagnostic returns
    NaN for ratios that do not imply a supersonic shock.
    """

    ratio = _as_float_array(ratio)
    mach = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 1.0)
    if not np.any(valid) or not np.isfinite(gamma) or gamma <= 1.0:
        return mach
    values = ratio[valid]
    a = 2.0 * gamma * (gamma - 1.0)
    b = 4.0 * gamma - (gamma - 1.0) ** 2 - values * (gamma + 1.0) ** 2
    c = -2.0 * (gamma - 1.0)
    discriminant = b * b - 4.0 * a * c
    good = np.isfinite(discriminant) & (discriminant >= 0.0)
    recovered = np.full(values.shape, np.nan, dtype=np.float64)
    m2 = np.full(values.shape, np.nan, dtype=np.float64)
    m2[good] = (-b[good] + np.sqrt(discriminant[good])) / (2.0 * a)
    supersonic = good & np.isfinite(m2) & (m2 > 1.0)
    recovered[supersonic] = np.sqrt(m2[supersonic])
    mach[valid] = recovered
    return mach


def mach_from_pressure_ratio(ratio, gamma: float = 5.0 / 3.0):
    """Return Mach number inferred from the thermal-pressure ratio."""

    ratio = _as_float_array(ratio)
    mach = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 1.0)
    if not np.any(valid) or not np.isfinite(gamma) or gamma <= 1.0:
        return mach
    m2 = ((gamma + 1.0) * ratio[valid] + (gamma - 1.0)) / (2.0 * gamma)
    good = np.isfinite(m2) & (m2 > 1.0)
    recovered = np.full(m2.shape, np.nan, dtype=np.float64)
    recovered[good] = np.sqrt(m2[good])
    mach[valid] = recovered
    return mach


def density_saturation_mask(
    ratio,
    gamma: float = 5.0 / 3.0,
    *,
    saturation_rtol: float = 1.0e-6,
):
    """Identify compression ratios too close to the strong-shock limit."""

    ratio = _as_float_array(ratio)
    if not np.isfinite(gamma) or gamma <= 1.0:
        return np.zeros(ratio.shape, dtype=bool)
    limit = (gamma + 1.0) / (gamma - 1.0)
    return (
        np.isfinite(ratio)
        & (ratio > 1.0)
        & (ratio >= limit * (1.0 - saturation_rtol))
    )


def mach_from_density_ratio(
    ratio,
    gamma: float = 5.0 / 3.0,
    *,
    saturation_rtol: float = 1.0e-6,
):
    """Return density-jump Mach, leaving saturated/invalid values as NaN.

    No clipping is performed at the strong-shock compression limit.
    """

    ratio = _as_float_array(ratio)
    mach = np.full(ratio.shape, np.nan, dtype=np.float64)
    if not np.isfinite(gamma) or gamma <= 1.0:
        return mach
    limit = (gamma + 1.0) / (gamma - 1.0)
    saturated = density_saturation_mask(
        ratio, gamma, saturation_rtol=saturation_rtol
    )
    valid = (
        np.isfinite(ratio)
        & (ratio > 1.0)
        & (ratio < limit)
        & ~saturated
    )
    denominator = (gamma + 1.0) - (gamma - 1.0) * ratio
    valid &= np.isfinite(denominator) & (denominator > 0.0)
    m2 = np.full(ratio.shape, np.nan, dtype=np.float64)
    np.divide(2.0 * ratio, denominator, out=m2, where=valid)
    valid &= np.isfinite(m2) & (m2 > 1.0)
    mach[valid] = np.sqrt(m2[valid])
    return mach


def mach_from_temperature_jump(upstream, downstream, gamma: float = 5.0 / 3.0):
    """Temperature estimator accepting endpoint states directly."""

    return mach_from_temperature_ratio(jump_ratio(upstream, downstream), gamma)


def mach_from_pressure_jump(upstream, downstream, gamma: float = 5.0 / 3.0):
    """Thermal-pressure estimator accepting endpoint states directly."""

    return mach_from_pressure_ratio(jump_ratio(upstream, downstream), gamma)


def mach_from_density_jump(
    upstream,
    downstream,
    gamma: float = 5.0 / 3.0,
    *,
    saturation_rtol: float = 1.0e-6,
):
    """Density estimator accepting endpoint states directly."""

    return mach_from_density_ratio(
        jump_ratio(upstream, downstream),
        gamma,
        saturation_rtol=saturation_rtol,
    )


__all__ = [
    "MachValidationFlag",
    "density_saturation_mask",
    "jump_ratio",
    "mach_from_density_jump",
    "mach_from_density_ratio",
    "mach_from_pressure_jump",
    "mach_from_pressure_ratio",
    "mach_from_temperature_jump",
    "mach_from_temperature_ratio",
]
