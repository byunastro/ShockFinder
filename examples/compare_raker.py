"""Compare a saved ShockFinder result with a RAKER ``shock_XXXXX.dat``."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np


KPC_IN_KM = 3.0856775814913673e16


def _open_raker(path, reader_path):
    spec = importlib.util.spec_from_file_location("raker_reader", reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import RAKER reader: {reader_path}")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.read(4)
        version = struct.unpack("<i", stream.read(4))[0]
        count = struct.unpack("<q", stream.read(8))[0]
        record_bytes = struct.unpack("<i", stream.read(4))[0]
    if magic != b"RKSC":
        raise ValueError(f"bad RAKER magic: {magic!r}")
    if record_bytes != reader.SHOCK_DTYPE.itemsize:
        raise ValueError(
            f"RAKER record size {record_bytes} != {reader.SHOCK_DTYPE.itemsize}"
        )
    if path.stat().st_size != 20 + count * record_bytes:
        raise ValueError("RAKER file size does not match its header")
    print(f"RAKER version={version}, cells={count:,}")
    return np.memmap(
        path, mode="r", dtype=reader.SHOCK_DTYPE, offset=20, shape=(count,)
    )


def compare_raker(
    result,
    raker_shock_path,
    *,
    snap_unit_kpc,
    reader_path="amr_data/RAKER/read.py",
    min_mach=1.0,
    verify_cells=True,
    allow_spatial_subset=True,
    chunk_size=1_000_000,
    position_tolerance_cells=1.0e-5,
):
    """Compare an existing ``finder.ShockFinder(cell)`` result with RAKER.

    RAKER positions are converted with ``position / snap.unit['kpc']``. If
    RAKER covers the full snapshot, its cells are streamed and restricted to
    the ShockFinder result's position bounds and AMR-level range.
    """

    raker = _open_raker(raker_shock_path, reader_path)
    if len(raker) != len(result.mach) and not allow_spatial_subset:
        raise ValueError(
            f"cell-count mismatch: ShockFinder={len(result.mach):,}, "
            f"RAKER={len(raker):,}"
        )
    if len(raker) != len(result.mach):
        return _subset_metrics(
            result,
            raker,
            snap_unit_kpc,
            min_mach,
            verify_cells,
            chunk_size,
            position_tolerance_cells,
        )
    if verify_cells:
        _verify_cells(
            result,
            raker,
            snap_unit_kpc,
            chunk_size,
            position_tolerance_cells,
        )
    return _metrics(result, raker, min_mach)


# Compatibility with notebooks that imported the earlier descriptive name.
compare_raker_result = compare_raker


def _verify_cells(result, raker, snap_unit_kpc, chunk_size, tolerance):
    if not np.isfinite(snap_unit_kpc) or snap_unit_kpc <= 0:
        raise ValueError("snap_unit_kpc must be positive and finite")
    if result.pos is None or result.dx is None or result.level is None:
        raise ValueError("ShockFinder result must contain pos, dx, and level")
    for start in range(0, len(raker), chunk_size):
        stop = min(start + chunk_size, len(raker))
        if not np.array_equal(raker["lv"][start:stop], result.level[start:stop]):
            raise ValueError(f"AMR level or row order differs near row {start:,}")
        raker_kpc = np.column_stack(
            [raker[name][start:stop] for name in ("x", "y", "z")]
        ) / snap_unit_kpc
        finder_kpc = np.asarray(result.pos[start:stop]) / KPC_IN_KM
        dx_kpc = np.asarray(result.dx[start:stop]) / KPC_IN_KM
        error = np.max(np.abs(raker_kpc - finder_kpc) / dx_kpc[:, None], axis=1)
        bad = ~np.isfinite(error) | (error > tolerance)
        if np.any(bad):
            row = start + int(np.flatnonzero(bad)[0])
            raise ValueError(f"cell position or row order differs at row {row:,}")


def _metrics(result, raker, min_mach):
    sf_mach = np.asarray(result.mach, dtype=np.float64)
    rk_mach = np.asarray(raker["Mach"], dtype=np.float64)
    sf = np.asarray(result.shock) & np.isfinite(sf_mach) & (sf_mach >= min_mach)
    rk = (raker["is_shock"] == 1) & np.isfinite(rk_mach) & (rk_mach >= min_mach)
    both = sf & rk
    union = sf | rk
    nsf = int(np.count_nonzero(sf))
    nrk = int(np.count_nonzero(rk))
    nboth = int(np.count_nonzero(both))
    nunion = int(np.count_nonzero(union))
    output = {
        "n_cells": int(len(raker)),
        "min_mach": float(min_mach),
        "shockfinder_shocks": nsf,
        "raker_shocks": nrk,
        "intersection": nboth,
        "union": nunion,
        "shockfinder_only": nsf - nboth,
        "raker_only": nrk - nboth,
        "precision_vs_raker": None if nsf == 0 else nboth / nsf,
        "recall_vs_raker": None if nrk == 0 else nboth / nrk,
        "jaccard": None if nunion == 0 else nboth / nunion,
    }
    if nboth:
        ratio = sf_mach[both] / rk_mach[both]
        output["paired_mach"] = {
            "count": nboth,
            "median_shockfinder_over_raker": float(np.median(ratio)),
            "within_10_percent": float(np.mean(np.abs(ratio - 1) <= 0.1)),
            "within_30_percent": float(np.mean(np.abs(ratio - 1) <= 0.3)),
        }
    else:
        output["paired_mach"] = {"count": 0}
    return output


def _subset_metrics(
    result,
    raker,
    snap_unit_kpc,
    min_mach,
    verify_cells,
    chunk_size,
    tolerance,
):
    """Stream a full-snapshot RAKER file and compare only the result box."""

    if not np.isfinite(snap_unit_kpc) or snap_unit_kpc <= 0:
        raise ValueError("snap_unit_kpc must be positive and finite")
    if result.pos is None or result.dx is None or result.level is None:
        raise ValueError("ShockFinder result must contain pos, dx, and level")

    # Work in RAMSES code coordinates while filtering to avoid repeatedly
    # converting the 15e8-row RAKER file to kpc.
    code_scale = snap_unit_kpc / KPC_IN_KM
    # Reduce one strided coordinate at a time; never materialize another
    # N-by-3 position array for a hundreds-of-millions-cell result.
    lower = np.array(
        [float(np.min(result.pos[:, axis])) * code_scale for axis in range(3)]
    )
    upper = np.array(
        [float(np.max(result.pos[:, axis])) * code_scale for axis in range(3)]
    )
    level_min = int(np.min(result.level))
    level_max = int(np.max(result.level))
    absolute_tolerance = (
        float(np.max(result.dx)) / KPC_IN_KM * snap_unit_kpc * tolerance
    )
    lower -= absolute_tolerance
    upper += absolute_tolerance

    result_row = 0
    nsf = nrk = nboth = nunion = 0
    ratio_parts = []
    print(
        "Selecting RAKER cells inside ShockFinder bounds: "
        f"levels={level_min}..{level_max}, total RAKER rows={len(raker):,}"
    )
    for start in range(0, len(raker), chunk_size):
        stop = min(start + chunk_size, len(raker))
        if start and (start // chunk_size) % 100 == 0:
            print(
                f"RAKER scan: {stop:,}/{len(raker):,} rows; "
                f"matched subset rows={result_row:,}"
            )
        block = raker[start:stop]
        selected_mask = (
            (block["lv"] >= level_min)
            & (block["lv"] <= level_max)
            & (block["x"] >= lower[0])
            & (block["x"] <= upper[0])
            & (block["y"] >= lower[1])
            & (block["y"] <= upper[1])
            & (block["z"] >= lower[2])
            & (block["z"] <= upper[2])
        )
        if not np.any(selected_mask):
            continue
        selected = block[selected_mask]
        next_row = result_row + len(selected)
        if next_row > len(result.mach):
            raise ValueError(
                "RAKER bounding-box selection contains more cells than the "
                "ShockFinder result. The ShockFinder selection may not be a "
                "rectangular box, or the level selections differ."
            )

        if verify_cells:
            expected_level = np.asarray(result.level[result_row:next_row])
            if not np.array_equal(selected["lv"], expected_level):
                raise ValueError(
                    f"RAKER and ShockFinder row order/levels differ near "
                    f"ShockFinder row {result_row:,}"
                )
            raker_kpc = np.column_stack(
                [selected[name] for name in ("x", "y", "z")]
            ) / snap_unit_kpc
            finder_kpc = np.asarray(result.pos[result_row:next_row]) / KPC_IN_KM
            dx_kpc = np.asarray(result.dx[result_row:next_row]) / KPC_IN_KM
            error = np.max(
                np.abs(raker_kpc - finder_kpc) / dx_kpc[:, None], axis=1
            )
            if np.any(~np.isfinite(error) | (error > tolerance)):
                bad = int(np.flatnonzero(~np.isfinite(error) | (error > tolerance))[0])
                raise ValueError(
                    f"RAKER and ShockFinder positions/order differ at "
                    f"ShockFinder row {result_row + bad:,}"
                )

        sf_mach = np.asarray(result.mach[result_row:next_row], dtype=np.float64)
        rk_mach = np.asarray(selected["Mach"], dtype=np.float64)
        sf = (
            np.asarray(result.shock[result_row:next_row])
            & np.isfinite(sf_mach)
            & (sf_mach >= min_mach)
        )
        rk = (
            (selected["is_shock"] == 1)
            & np.isfinite(rk_mach)
            & (rk_mach >= min_mach)
        )
        both = sf & rk
        nsf += int(np.count_nonzero(sf))
        nrk += int(np.count_nonzero(rk))
        nboth += int(np.count_nonzero(both))
        nunion += int(np.count_nonzero(sf | rk))
        if np.any(both):
            ratio_parts.append(sf_mach[both] / rk_mach[both])
        result_row = next_row

    if result_row != len(result.mach):
        raise ValueError(
            f"RAKER bounds selected {result_row:,} cells, but ShockFinder has "
            f"{len(result.mach):,}. Ensure both use the same box and levels."
        )
    output = {
        "raker_full_snapshot_cells": int(len(raker)),
        "n_cells": int(result_row),
        "comparison_mode": "spatial_subset_stream",
        "min_mach": float(min_mach),
        "shockfinder_shocks": nsf,
        "raker_shocks": nrk,
        "intersection": nboth,
        "union": nunion,
        "shockfinder_only": nsf - nboth,
        "raker_only": nrk - nboth,
        "precision_vs_raker": None if nsf == 0 else nboth / nsf,
        "recall_vs_raker": None if nrk == 0 else nboth / nrk,
        "jaccard": None if nunion == 0 else nboth / nunion,
    }
    if ratio_parts:
        ratios = np.concatenate(ratio_parts)
        output["paired_mach"] = {
            "count": nboth,
            "median_shockfinder_over_raker": float(np.median(ratios)),
            "within_10_percent": float(np.mean(np.abs(ratios - 1) <= 0.1)),
            "within_30_percent": float(np.mean(np.abs(ratios - 1) <= 0.3)),
        }
    else:
        output["paired_mach"] = {"count": 0}
    return output
