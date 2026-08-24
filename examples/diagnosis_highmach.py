"""Inspect the cells that form the high-Mach tail of a saved ShockResult.

This script deliberately separates structural checks, which can be performed
from ``ShockResult`` alone, from thermodynamic Rankine-Hugoniot checks, which
require the original AMR cell data and are therefore not attempted here.

Example
-------
python examples/diagnosis_highmach.py

python examples/diagnosis_highmach.py \
    --result /storage1/byunkh/NC_map/shockmap/new_result_00785_lmin13_lmax19_Tmin1e5.pkl \
    --min-mach 40 --output-dir highmach_diagnosis
"""

from __future__ import annotations

import argparse
import csv
import importlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT = Path(
    "/storage1/byunkh/NC_map/shockmap/"
    "new_result_00785_lmin13_lmax19_Tmin1e5.pkl"
)

K_BOLTZMANN = 1.380649e-16  # erg K^-1
PROTON_MASS = 1.67262192369e-24  # g


@dataclass(slots=True)
class RHDiagnosis:
    """Rankine-Hugoniot checks, with one row per selected high-Mach shock.

    ``result_rows`` index the retained-cell arrays in ``result`` and
    ``original_cell_indices`` index the original ``cell`` arrays.
    """

    result_rows: np.ndarray
    original_cell_indices: np.ndarray
    mach_temperature: np.ndarray
    mach_velocity: np.ndarray
    upstream_temperature: np.ndarray
    downstream_temperature: np.ndarray
    upstream_density: np.ndarray
    downstream_density: np.ndarray
    temperature_ratio: np.ndarray
    compression_observed: np.ndarray
    compression_expected: np.ndarray
    pressure_ratio_observed: np.ndarray
    pressure_ratio_expected: np.ndarray
    shock_speed_normal: np.ndarray
    upstream_speed_shock_frame: np.ndarray
    downstream_speed_shock_frame: np.ndarray
    density_log_residual: np.ndarray
    mach_log_residual: np.ndarray
    mass_flux_log_residual: np.ndarray
    entropy_log_jump: np.ndarray
    zone_width_over_dx: np.ndarray
    level_span: np.ndarray
    valid_endpoints: np.ndarray
    flag_density_inconsistent: np.ndarray
    flag_mach_inconsistent: np.ndarray
    flag_mass_flux_inconsistent: np.ndarray
    flag_entropy_nonincrease: np.ndarray
    flag_long_zone: np.ndarray
    flag_large_level_jump: np.ndarray
    flag_invalid: np.ndarray
    flag_suspect: np.ndarray

    def summary(self) -> dict[str, int | float]:
        count = self.result_rows.size
        return {
            "tested_cells": int(count),
            "suspect_cells": int(np.count_nonzero(self.flag_suspect)),
            "suspect_fraction": (
                float(np.count_nonzero(self.flag_suspect) / count)
                if count else np.nan
            ),
            "density_inconsistent": int(np.count_nonzero(self.flag_density_inconsistent)),
            "mach_inconsistent": int(np.count_nonzero(self.flag_mach_inconsistent)),
            "mass_flux_inconsistent": int(np.count_nonzero(self.flag_mass_flux_inconsistent)),
            "entropy_nonincrease": int(np.count_nonzero(self.flag_entropy_nonincrease)),
            "long_zone": int(np.count_nonzero(self.flag_long_zone)),
            "large_level_jump": int(np.count_nonzero(self.flag_large_level_jump)),
            "invalid": int(np.count_nonzero(self.flag_invalid)),
        }


def _cell_field(cell, name: str, unit: str) -> np.ndarray:
    """Read a unit-aware cell field, falling back to a plain field lookup."""

    try:
        value = cell[name, unit]
    except (KeyError, TypeError, ValueError, IndexError):
        value = cell[name]
    return np.asarray(value, dtype=np.float64)


def rh_diagnosis(
    cell,
    result,
    *,
    min_mach: float = 40.0,
    gamma: float = 5.0 / 3.0,
    mu: float = 0.59,
    temperature_floor: float = 1.0e4,
    density_log_tolerance: float = 0.4,
    mach_log_tolerance: float = np.log(2.0),
    mass_flux_log_tolerance: float = 0.2,
    zone_ratio_limit: float = 8.0,
    level_jump_limit: int = 1,
) -> RHDiagnosis:
    """Diagnose high-Mach cells using independent jump-condition checks.

    Temperature-derived Mach numbers in ``result`` are compared with the
    density jump and with a velocity-derived Mach number.  The latter estimates
    the shock speed from mass conservation in the endpoint normal direction.

    The default tolerances are screening thresholds, not universal physical
    cuts. Radiative, magnetized, cosmic-ray-modified, or broadened shocks may
    legitimately fail ideal-gas hydrodynamic Rankine-Hugoniot conditions.
    """

    if gamma <= 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and greater than 1")
    if min_mach <= 1.0:
        raise ValueError("min_mach must be greater than 1")

    mach_all = np.asarray(result.mach, dtype=np.float64)
    shock_all = np.asarray(result.shock, dtype=bool)
    n = mach_all.size
    upstream_all = optional_array(result, "upstream_index", n, dtype=np.int64, fill=-1)
    downstream_all = optional_array(result, "downstream_index", n, dtype=np.int64, fill=-1)
    selected = optional_array(result, "selected_indices", n, dtype=np.int64, fill=-1)
    dx = optional_array(result, "dx", n)
    level = optional_array(result, "level", n)
    zone_width = optional_array(result, "zone_width", n)

    choose = shock_all & np.isfinite(mach_all) & (mach_all >= min_mach)
    rows = np.nonzero(choose)[0]
    mach = mach_all[rows]
    upstream = upstream_all[rows]
    downstream = downstream_all[rows]
    valid_endpoints = (
        (upstream >= 0) & (upstream < n) &
        (downstream >= 0) & (downstream < n)
    )

    # Endpoint arrays use retained-cell indices; cell fields use original rows.
    upstream_original = np.full(rows.size, -1, dtype=np.int64)
    downstream_original = np.full(rows.size, -1, dtype=np.int64)
    upstream_original[valid_endpoints] = selected[upstream[valid_endpoints]]
    downstream_original[valid_endpoints] = selected[downstream[valid_endpoints]]
    original = selected[rows]

    temperature = _cell_field(cell, "T", "K")
    density = _cell_field(cell, "rho", "Msol/kpc3")
    vx = _cell_field(cell, "vx", "km/s")
    vy = _cell_field(cell, "vy", "km/s")
    vz = _cell_field(cell, "vz", "km/s")
    field_size = temperature.size
    if any(field.size != field_size for field in (density, vx, vy, vz)):
        raise ValueError("cell temperature, density, and velocity fields differ in length")
    valid_original = (
        valid_endpoints & (upstream_original >= 0) & (upstream_original < field_size)
        & (downstream_original >= 0) & (downstream_original < field_size)
    )

    def endpoint_field(field: np.ndarray, original_indices: np.ndarray) -> np.ndarray:
        output = np.full(rows.size, np.nan)
        output[valid_original] = field[original_indices[valid_original]]
        return output

    temp1_raw = endpoint_field(temperature, upstream_original)
    temp2 = endpoint_field(temperature, downstream_original)
    temp1 = np.maximum(temp1_raw, temperature_floor)
    rho1 = endpoint_field(density, upstream_original)
    rho2 = endpoint_field(density, downstream_original)
    vel1 = np.column_stack((
        endpoint_field(vx, upstream_original), endpoint_field(vy, upstream_original),
        endpoint_field(vz, upstream_original),
    ))
    vel2 = np.column_stack((
        endpoint_field(vx, downstream_original), endpoint_field(vy, downstream_original),
        endpoint_field(vz, downstream_original),
    ))

    normal = np.full((rows.size, 3), np.nan)
    stored_normal = getattr(result, "normal", None)
    if stored_normal is not None:
        normal[:] = np.asarray(stored_normal, dtype=np.float64)[rows]
    elif getattr(result, "pos", None) is not None:
        pos = np.asarray(result.pos, dtype=np.float64)
        normal[valid_endpoints] = (
            pos[downstream[valid_endpoints]] - pos[upstream[valid_endpoints]]
        )
    normal_norm = np.linalg.norm(normal, axis=1)
    valid_normal = np.isfinite(normal_norm) & (normal_norm > 0.0)
    normal[valid_normal] /= normal_norm[valid_normal, None]

    vn1 = np.einsum("ij,ij->i", vel1, normal)
    vn2 = np.einsum("ij,ij->i", vel2, normal)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        temperature_ratio = temp2 / temp1
        compression_observed = rho2 / rho1
        m2 = mach * mach
        compression_expected = ((gamma + 1.0) * m2) / ((gamma - 1.0) * m2 + 2.0)
        pressure_ratio_observed = compression_observed * temperature_ratio
        pressure_ratio_expected = (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)

        # rho1(v1-vs) = rho2(v2-vs), projected along the endpoint normal.
        shock_speed = (rho2 * vn2 - rho1 * vn1) / (rho2 - rho1)
        u1 = vn1 - shock_speed
        u2 = vn2 - shock_speed
        sound_speed = np.sqrt(
            gamma * K_BOLTZMANN * temp1 / (mu * PROTON_MASS)
        ) / 1.0e5
        mach_velocity = np.abs(u1) / sound_speed

        density_residual = np.abs(np.log(compression_observed / compression_expected))
        mach_residual = np.abs(np.log(mach / mach_velocity))
        mass_flux_residual = np.abs(np.log(np.abs(rho1 * u1) / np.abs(rho2 * u2)))
        entropy_log_jump = np.log(
            (temp2 / rho2 ** (gamma - 1.0)) /
            (temp1 / rho1 ** (gamma - 1.0))
        )

    zone_ratio = np.full(rows.size, np.nan)
    good_dx = np.isfinite(dx[rows]) & (dx[rows] > 0.0)
    zone_ratio[good_dx] = zone_width[rows][good_dx] / dx[rows][good_dx]
    upstream_level = np.full(rows.size, np.nan)
    downstream_level = np.full(rows.size, np.nan)
    upstream_level[valid_endpoints] = level[upstream[valid_endpoints]]
    downstream_level[valid_endpoints] = level[downstream[valid_endpoints]]
    center_level = level[rows]
    level_max = np.fmax(np.fmax(center_level, upstream_level), downstream_level)
    level_min = np.fmin(np.fmin(center_level, upstream_level), downstream_level)
    level_span = level_max - level_min

    finite_physics = (
        valid_original & valid_normal & np.isfinite(temp1) & (temp1 > 0.0)
        & np.isfinite(temp2) & (temp2 > 0.0) & np.isfinite(rho1) & (rho1 > 0.0)
        & np.isfinite(rho2) & (rho2 > 0.0)
    )
    invalid = ~finite_physics | ~np.isfinite(mach_velocity)
    density_bad = finite_physics & (
        ~np.isfinite(density_residual) | (density_residual > density_log_tolerance)
    )
    mach_bad = finite_physics & (
        ~np.isfinite(mach_residual) | (mach_residual > mach_log_tolerance)
    )
    mass_bad = finite_physics & (
        ~np.isfinite(mass_flux_residual) | (mass_flux_residual > mass_flux_log_tolerance)
    )
    entropy_bad = finite_physics & (
        ~np.isfinite(entropy_log_jump) | (entropy_log_jump <= 0.0)
    )
    long_zone = np.isfinite(zone_ratio) & (zone_ratio > zone_ratio_limit)
    level_bad = np.isfinite(level_span) & (level_span > level_jump_limit)
    suspect = invalid | density_bad | mach_bad | mass_bad | entropy_bad | long_zone | level_bad

    return RHDiagnosis(
        result_rows=rows, original_cell_indices=original,
        mach_temperature=mach, mach_velocity=mach_velocity,
        upstream_temperature=temp1_raw, downstream_temperature=temp2,
        upstream_density=rho1, downstream_density=rho2,
        temperature_ratio=temperature_ratio,
        compression_observed=compression_observed,
        compression_expected=compression_expected,
        pressure_ratio_observed=pressure_ratio_observed,
        pressure_ratio_expected=pressure_ratio_expected,
        shock_speed_normal=shock_speed,
        upstream_speed_shock_frame=u1, downstream_speed_shock_frame=u2,
        density_log_residual=density_residual, mach_log_residual=mach_residual,
        mass_flux_log_residual=mass_flux_residual,
        entropy_log_jump=entropy_log_jump, zone_width_over_dx=zone_ratio,
        level_span=level_span, valid_endpoints=valid_endpoints,
        flag_density_inconsistent=density_bad, flag_mach_inconsistent=mach_bad,
        flag_mass_flux_inconsistent=mass_bad, flag_entropy_nonincrease=entropy_bad,
        flag_long_zone=long_zone, flag_large_level_jump=level_bad,
        flag_invalid=invalid, flag_suspect=suspect,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--min-mach", type=float, default=40.0,
        help="Mach threshold defining the tail (default: 40)",
    )
    parser.add_argument(
        "--loader", default="rur.utool",
        help="module containing load(path), or 'pickle' (default: rur.utool)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("highmach_diagnosis"),
    )
    parser.add_argument(
        "--zone-ratio-limit", type=float, default=8.0,
        help="flag zone_width/dx_center above this value (default: 8)",
    )
    parser.add_argument(
        "--level-jump-limit", type=int, default=1,
        help="flag endpoint level jumps larger than this value (default: 1)",
    )
    return parser.parse_args()


def make_loader(name: str):
    if name == "pickle":
        def load_pickle(path: Path):
            with path.open("rb") as stream:
                return pickle.load(stream)
        return load_pickle

    module = importlib.import_module(name)
    if not hasattr(module, "load"):
        raise AttributeError(f"{name!r} has no load(path) function")
    return lambda path: module.load(str(path))


def optional_array(result, name: str, n: int, *, dtype=float, fill=np.nan):
    value = getattr(result, name, None)
    if value is None:
        return np.full(n, fill, dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if array.shape[0] != n:
        raise ValueError(f"result.{name} has length {array.shape[0]}, expected {n}")
    return array


def finite_percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else np.nan


def endpoint_values(values: np.ndarray, indices: np.ndarray, valid: np.ndarray):
    output = np.full(indices.size, np.nan, dtype=np.float64)
    output[valid] = values[indices[valid]]
    return output


def analyse(result, min_mach: float, zone_ratio_limit: float, level_jump_limit: int):
    mach = np.asarray(result.mach, dtype=np.float64)
    shock = np.asarray(result.shock, dtype=bool)
    n = mach.size
    if shock.shape != mach.shape:
        raise ValueError("result.shock and result.mach must have the same shape")

    upstream = optional_array(result, "upstream_index", n, dtype=np.int64, fill=-1)
    downstream = optional_array(result, "downstream_index", n, dtype=np.int64, fill=-1)
    center = optional_array(result, "center_index", n, dtype=np.int64, fill=-1)
    selected = optional_array(result, "selected_indices", n, dtype=np.int64, fill=-1)
    dx = optional_array(result, "dx", n)
    level = optional_array(result, "level", n)
    zone_width = optional_array(result, "zone_width", n)

    valid_up = (upstream >= 0) & (upstream < n)
    valid_down = (downstream >= 0) & (downstream < n)
    valid_endpoints = valid_up & valid_down

    upstream_level = endpoint_values(level, upstream, valid_up)
    downstream_level = endpoint_values(level, downstream, valid_down)
    # np.fmax/fmin preserve NaN only when every compared value is missing and
    # avoid all-NaN-slice warnings for results saved before ``level`` existed.
    level_max = np.fmax(np.fmax(level, upstream_level), downstream_level)
    level_min = np.fmin(np.fmin(level, upstream_level), downstream_level)
    level_span = level_max - level_min

    zone_ratio = np.full(n, np.nan)
    positive_dx = np.isfinite(dx) & (dx > 0.0)
    valid_zone = positive_dx & np.isfinite(zone_width) & (zone_width >= 0.0)
    zone_ratio[valid_zone] = zone_width[valid_zone] / dx[valid_zone]

    pos = getattr(result, "pos", None)
    endpoint_distance = np.full(n, np.nan)
    endpoint_dx_min = np.full(n, np.nan)
    if pos is not None:
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape != (n, 3):
            raise ValueError(f"result.pos has shape {pos.shape}, expected {(n, 3)}")
        rows = np.nonzero(valid_endpoints)[0]
        endpoint_distance[rows] = np.linalg.norm(
            pos[downstream[rows]] - pos[upstream[rows]], axis=1
        )
        endpoint_scale = np.minimum(dx[upstream[rows]], dx[downstream[rows]])
        ok = np.isfinite(endpoint_scale) & (endpoint_scale > 0.0)
        endpoint_dx_min[rows[ok]] = endpoint_distance[rows[ok]] / endpoint_scale[ok]

    finite_shock = shock & np.isfinite(mach) & (mach > 0.0)
    high = finite_shock & (mach >= min_mach)
    bad_index = high & ~valid_endpoints
    long_zone = high & np.isfinite(zone_ratio) & (zone_ratio > zone_ratio_limit)
    large_level_jump = high & np.isfinite(level_span) & (level_span > level_jump_limit)
    width_mismatch = (
        high & np.isfinite(zone_width) & np.isfinite(endpoint_distance)
        & ~np.isclose(zone_width, endpoint_distance, rtol=1.0e-8, atol=0.0)
    )
    suspect = bad_index | long_zone | large_level_jump | width_mismatch

    rows = np.nonzero(high)[0]
    columns = {
        "result_row": rows,
        "original_cell_index": selected[rows],
        "mach": mach[rows],
        "center_index": center[rows],
        "upstream_index": upstream[rows],
        "downstream_index": downstream[rows],
        "level_center": level[rows],
        "level_upstream": upstream_level[rows],
        "level_downstream": downstream_level[rows],
        "level_span": level_span[rows],
        "dx_center": dx[rows],
        "zone_width": zone_width[rows],
        "zone_width_over_dx": zone_ratio[rows],
        "endpoint_distance": endpoint_distance[rows],
        "endpoint_distance_over_min_endpoint_dx": endpoint_dx_min[rows],
        "flag_invalid_endpoint": bad_index[rows],
        "flag_long_zone": long_zone[rows],
        "flag_large_level_jump": large_level_jump[rows],
        "flag_width_mismatch": width_mismatch[rows],
        "flag_suspect": suspect[rows],
    }
    if pos is not None:
        columns.update({
            "x": pos[rows, 0], "y": pos[rows, 1], "z": pos[rows, 2],
        })

    summary = {
        "retained_cells": n,
        "finite_shock_cells": int(np.count_nonzero(finite_shock)),
        "high_mach_cells": int(rows.size),
        "high_mach_fraction_of_shocks": (
            float(rows.size / np.count_nonzero(finite_shock))
            if np.any(finite_shock) else np.nan
        ),
        "mach_max": float(np.max(mach[finite_shock])) if np.any(finite_shock) else np.nan,
        "high_mach_median": finite_percentile(mach[high], 50),
        "high_mach_zone_ratio_median": finite_percentile(zone_ratio[high], 50),
        "high_mach_zone_ratio_p90": finite_percentile(zone_ratio[high], 90),
        "flag_invalid_endpoint": int(np.count_nonzero(bad_index)),
        "flag_long_zone": int(np.count_nonzero(long_zone)),
        "flag_large_level_jump": int(np.count_nonzero(large_level_jump)),
        "flag_width_mismatch": int(np.count_nonzero(width_mismatch)),
        "flag_suspect_union": int(np.count_nonzero(suspect)),
    }
    return columns, summary, mach[finite_shock], zone_ratio[finite_shock], high[finite_shock]


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    count = len(columns[names[0]]) if names else 0
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for i in range(count):
            writer.writerow([columns[name][i] for name in names])


def write_summary(path: Path, summary: dict[str, object], result) -> None:
    diagnostics = getattr(result, "diagnostics", None)
    with path.open("w") as stream:
        for key, value in summary.items():
            stream.write(f"{key}: {value}\n")
        if diagnostics:
            stream.write("\nShockFinder run diagnostics:\n")
            for key, value in diagnostics.items():
                stream.write(f"  {key}: {value}\n")
        stream.write(
            "\nImportant: structural flags are screening tests, not proof that a "
            "shock is unphysical. Density, temperature, and velocity endpoint "
            "fields are required for an independent Rankine-Hugoniot test.\n"
        )


def make_plot(
    path: Path, mach: np.ndarray, zone_ratio: np.ndarray,
    high_in_shocks: np.ndarray, min_mach: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if mach.size:
        lo = max(1.0, float(np.min(mach)))
        hi = max(lo * 1.01, float(np.max(mach)) * 1.001)
        edges = np.logspace(np.log10(lo), np.log10(hi), 61)
        axes[0].hist(mach, bins=edges, histtype="step", linewidth=1.5)
    axes[0].axvline(min_mach, color="tab:red", linestyle="--", label="tail threshold")
    axes[0].set(xscale="log", yscale="log", xlabel="Mach", ylabel="shock-cell count")
    axes[0].legend()

    ok = np.isfinite(zone_ratio) & (zone_ratio > 0.0)
    axes[1].scatter(mach[ok & ~high_in_shocks], zone_ratio[ok & ~high_in_shocks],
                    s=3, alpha=0.15, label="lower Mach")
    axes[1].scatter(mach[ok & high_in_shocks], zone_ratio[ok & high_in_shocks],
                    s=12, alpha=0.7, label="high Mach")
    axes[1].set(xscale="log", yscale="log", xlabel="Mach",
                ylabel="zone width / center dx")
    axes[1].legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.min_mach <= 1.0:
        raise ValueError("--min-mach must be greater than 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = make_loader(args.loader)(args.result)
    columns, summary, mach, zone_ratio, high = analyse(
        result, args.min_mach, args.zone_ratio_limit, args.level_jump_limit
    )
    write_csv(args.output_dir / "highmach_cells.csv", columns)
    write_summary(args.output_dir / "summary.txt", summary, result)
    make_plot(args.output_dir / "highmach_diagnostics.png", mach, zone_ratio, high,
              args.min_mach)

    print(f"Loaded: {args.result}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
