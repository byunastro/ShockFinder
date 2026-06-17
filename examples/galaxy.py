"""Classify galaxies by whether their trajectories crossed shock fronts.

This example assumes:
- ``cell`` is the AMR gas cell table used by ``shocktest.ShockFinder``.
- ``galaxy_pos_prev`` and ``galaxy_pos_now`` are ``(ngal, 3)`` arrays in km.
- The same galaxy order is used at the previous and current snapshots.

The classification is geometric: a galaxy is marked as crossed when its segment
between two snapshots changes sign across the nearest shock plane and passes
close enough to that shock cell.
"""

from __future__ import annotations

import numpy as np

import shocktest
from shocktest import pyShockFinder


KPC_IN_KM = 3.0856775814913673e16


def run_shockfinder(cell, *, minlevel=15, maxlevel=20, min_mach=1.5, show_progress=True):
    """Run ShockFinder and return the shock result plus dissipation fields."""

    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    finder.min_mach = min_mach
    finder.show_progress = show_progress

    result = finder.ShockFinder(cell)
    dissipation = pyShockFinder.compute_dissipation(cell, result)
    return result, dissipation


def shock_front_catalog(result, dissipation, *, min_mach=1.5, min_flux=0.0):
    """Build positions, normals, and strengths for selected shock cells."""

    shock_mask = result.shock & (result.mach >= min_mach) & (dissipation.flux > min_flux)
    shock_rows = np.nonzero(shock_mask)[0]

    shock_pos = result.pos[shock_rows]
    shock_dx = result.dx[shock_rows]
    shock_mach = result.mach[shock_rows]
    shock_flux = dissipation.flux[shock_rows]

    upstream = result.upstream_index[shock_rows]
    downstream = result.downstream_index[shock_rows]
    valid = (upstream >= 0) & (downstream >= 0)

    normal = np.zeros_like(shock_pos)
    normal[valid] = result.pos[downstream[valid]] - result.pos[upstream[valid]]
    norm = np.linalg.norm(normal, axis=1)
    valid &= norm > 0.0
    normal[valid] /= norm[valid, None]

    return {
        "rows": shock_rows,
        "pos": shock_pos,
        "dx": shock_dx,
        "mach": shock_mach,
        "flux": shock_flux,
        "normal": normal,
        "valid_normal": valid,
    }


def classify_galaxy_shock_crossing(
    galaxy_pos_prev,
    galaxy_pos_now,
    shock_catalog,
    *,
    search_radius_km=100.0 * KPC_IN_KM,
    width_factor=2.0,
):
    """Classify whether each galaxy crossed a nearby shock plane.

    Parameters
    ----------
    galaxy_pos_prev, galaxy_pos_now:
        Galaxy positions at two snapshots, in km, with matching row order.
    shock_catalog:
        Output from ``shock_front_catalog``.
    search_radius_km:
        Maximum distance from the current galaxy position to a shock cell.
    width_factor:
        Allowed transverse distance in units of the local shock-cell ``dx``.
    """

    galaxy_pos_prev = np.asarray(galaxy_pos_prev, dtype=np.float64)
    galaxy_pos_now = np.asarray(galaxy_pos_now, dtype=np.float64)
    if galaxy_pos_prev.shape != galaxy_pos_now.shape or galaxy_pos_now.shape[1] != 3:
        raise ValueError("galaxy positions must both have shape (ngal, 3)")

    shock_pos = shock_catalog["pos"]
    shock_normal = shock_catalog["normal"]
    valid_normal = shock_catalog["valid_normal"]
    if shock_pos.size == 0:
        return _empty_classification(galaxy_pos_now.shape[0])

    nearest, distance = _nearest_shock(galaxy_pos_now, shock_pos)
    near = (distance <= search_radius_km) & valid_normal[nearest]

    crossed = np.zeros(galaxy_pos_now.shape[0], dtype=bool)
    signed_prev = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    signed_now = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    transverse = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)

    if np.any(near):
        shock_idx = nearest[near]
        p0 = galaxy_pos_prev[near]
        p1 = galaxy_pos_now[near]
        xs = shock_pos[shock_idx]
        ns = shock_normal[shock_idx]

        signed_prev[near] = np.sum((p0 - xs) * ns, axis=1)
        signed_now[near] = np.sum((p1 - xs) * ns, axis=1)

        segment_mid = 0.5 * (p0 + p1)
        offset = segment_mid - xs
        normal_offset = np.sum(offset * ns, axis=1)[:, None] * ns
        transverse[near] = np.linalg.norm(offset - normal_offset, axis=1)

        local_width = width_factor * shock_catalog["dx"][shock_idx]
        crossed[near] = (signed_prev[near] * signed_now[near] <= 0.0) & (transverse[near] <= local_width)

    return {
        "crossed": crossed,
        "near_shock": near,
        "nearest_shock_row": np.where(near, shock_catalog["rows"][nearest], -1),
        "nearest_mach": np.where(near, shock_catalog["mach"][nearest], np.nan),
        "nearest_flux": np.where(near, shock_catalog["flux"][nearest], np.nan),
        "distance_to_shock": np.where(near, distance, np.nan),
        "signed_distance_prev": signed_prev,
        "signed_distance_now": signed_now,
        "transverse_distance": transverse,
    }


def _nearest_shock(points, shock_pos):
    """Return nearest shock index and distance for each point."""

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nearest_shock_numpy(points, shock_pos)

    distance, nearest = cKDTree(shock_pos).query(points, workers=-1)
    return nearest.astype(np.int64), distance


def _nearest_shock_numpy(points, shock_pos, chunk_size=4096):
    """Numpy fallback for systems without SciPy."""

    nearest = np.empty(points.shape[0], dtype=np.int64)
    distance = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        delta = points[start:stop, None, :] - shock_pos[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)
        nearest[start:stop] = np.argmin(dist2, axis=1)
        distance[start:stop] = np.sqrt(dist2[np.arange(stop - start), nearest[start:stop]])
    return nearest, distance


def _empty_classification(n_galaxies):
    return {
        "crossed": np.zeros(n_galaxies, dtype=bool),
        "near_shock": np.zeros(n_galaxies, dtype=bool),
        "nearest_shock_row": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_mach": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_flux": np.full(n_galaxies, np.nan, dtype=np.float64),
        "distance_to_shock": np.full(n_galaxies, np.nan, dtype=np.float64),
        "signed_distance_prev": np.full(n_galaxies, np.nan, dtype=np.float64),
        "signed_distance_now": np.full(n_galaxies, np.nan, dtype=np.float64),
        "transverse_distance": np.full(n_galaxies, np.nan, dtype=np.float64),
    }


if __name__ == "__main__":
    # Replace these with your simulation data.
    cell = ...
    galaxy_pos_prev = ...
    galaxy_pos_now = ...

    result, dissipation = run_shockfinder(cell)
    catalog = shock_front_catalog(result, dissipation, min_mach=1.5)
    classification = classify_galaxy_shock_crossing(
        galaxy_pos_prev,
        galaxy_pos_now,
        catalog,
        search_radius_km=100.0 * KPC_IN_KM,
    )

    print("N galaxies near shocks:", np.count_nonzero(classification["near_shock"]))
    print("N galaxies crossed shocks:", np.count_nonzero(classification["crossed"]))
