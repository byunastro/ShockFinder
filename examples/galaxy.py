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

import gc

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


def filter_large_shock_fronts(
    shock_catalog,
    *,
    link_length_km=50.0 * KPC_IN_KM,
    min_cells=30,
    min_extent_km=300.0 * KPC_IN_KM,
    min_total_flux=0.0,
):
    """Keep spatially extended shock components.

    This removes compact galaxy-scale SN/AGN shock islands before matching
    galaxies to shocks. Tune ``min_extent_km`` to the smallest merger-shock
    length scale you want to keep.
    """

    shock_pos = np.asarray(shock_catalog["pos"], dtype=np.float64)
    if shock_pos.size == 0:
        out = {key: value.copy() for key, value in shock_catalog.items()}
        out["component_id"] = np.empty(0, dtype=np.int64)
        out["component_size"] = np.empty(0, dtype=np.int64)
        out["component_extent"] = np.empty(0, dtype=np.float64)
        out["component_total_flux"] = np.empty(0, dtype=np.float64)
        return out

    labels = _connected_components(shock_pos, link_length_km)
    n_components = int(labels.max()) + 1

    component_size = np.bincount(labels, minlength=n_components).astype(np.int64)
    component_total_flux = np.bincount(
        labels,
        weights=np.asarray(shock_catalog["flux"], dtype=np.float64),
        minlength=n_components,
    )
    component_extent = np.zeros(n_components, dtype=np.float64)
    for component in range(n_components):
        pos = shock_pos[labels == component]
        component_extent[component] = _component_extent(pos)

    keep_component = (
        (component_size >= min_cells)
        & (component_extent >= min_extent_km)
        & (component_total_flux >= min_total_flux)
    )
    keep = keep_component[labels]

    out = {}
    for key, value in shock_catalog.items():
        arr = np.asarray(value)
        out[key] = arr[keep]
    out["component_id"] = labels[keep]
    out["component_size"] = component_size[labels[keep]]
    out["component_extent"] = component_extent[labels[keep]]
    out["component_total_flux"] = component_total_flux[labels[keep]]
    return out


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
        "nearest_component_id": _catalog_lookup(shock_catalog, "component_id", nearest, near, -1),
        "nearest_component_size": _catalog_lookup(shock_catalog, "component_size", nearest, near, -1),
        "nearest_component_extent": _catalog_lookup(shock_catalog, "component_extent", nearest, near, np.nan),
        "nearest_mach": np.where(near, shock_catalog["mach"][nearest], np.nan),
        "nearest_flux": np.where(near, shock_catalog["flux"][nearest], np.nan),
        "distance_to_shock": np.where(near, distance, np.nan),
        "signed_distance_prev": signed_prev,
        "signed_distance_now": signed_now,
        "transverse_distance": transverse,
    }


def compact_classification_results(classification, *, keep="near_or_crossed", keep_keys=None):
    """Copy only useful galaxy-classification rows into a compact result dict.

    ``keep="near_or_crossed"`` keeps galaxies flagged by either ``near_shock``
    or ``crossed`` and records their original row numbers as ``galaxy_index``.
    Use ``keep="crossed"`` for the smallest post-analysis table.
    """

    if keep_keys is None:
        keep_keys = tuple(classification)

    n_galaxies = _classification_length(classification)
    keep_mask = _classification_keep_mask(classification, keep, n_galaxies)
    compact = {
        "galaxy_index": np.nonzero(keep_mask)[0].astype(np.int64, copy=False),
        "n_galaxies": np.array(n_galaxies, dtype=np.int64),
    }

    for key in keep_keys:
        value = classification[key]
        arr = np.asarray(value)
        if arr.shape[:1] == (n_galaxies,):
            compact[key] = arr[keep_mask].copy()
        else:
            compact[key] = arr.copy()
    return compact


def finalize_shock_classification(
    classification,
    *,
    result=None,
    dissipation=None,
    shock_catalog=None,
    keep="near_or_crossed",
    keep_keys=None,
):
    """Return compact classification results and release large shock arrays."""

    compact = compact_classification_results(classification, keep=keep, keep_keys=keep_keys)
    release_shock_work_arrays(result, dissipation, shock_catalog, classification)
    return compact


def release_shock_work_arrays(*objects):
    """Release arrays from temporary shock-finding objects and dictionaries."""

    for obj in objects:
        if obj is None:
            continue
        clear = getattr(obj, "clear", None)
        if callable(clear):
            clear()
        elif isinstance(obj, dict):
            obj.clear()
    gc.collect()


def _connected_components(points, link_length):
    """Return component labels for points connected within ``link_length``."""

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _connected_components_numpy(points, link_length)

    tree = cKDTree(points)
    neighbors = tree.query_ball_tree(tree, link_length)
    return _label_neighbors(neighbors)


def _connected_components_numpy(points, link_length, chunk_size=2048):
    neighbors = [[] for _ in range(points.shape[0])]
    link2 = link_length * link_length
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        delta = points[start:stop, None, :] - points[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)
        rows, cols = np.nonzero(dist2 <= link2)
        for row, col in zip(rows, cols):
            neighbors[start + row].append(int(col))
    return _label_neighbors(neighbors)


def _label_neighbors(neighbors):
    labels = np.full(len(neighbors), -1, dtype=np.int64)
    label = 0
    for seed in range(len(neighbors)):
        if labels[seed] >= 0:
            continue
        labels[seed] = label
        stack = [seed]
        while stack:
            node = stack.pop()
            for other in neighbors[node]:
                if labels[other] < 0:
                    labels[other] = label
                    stack.append(other)
        label += 1
    return labels


def _component_extent(points):
    if points.shape[0] <= 1:
        return 0.0
    centered = points - np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return 0.0
    projected = centered @ vh[0]
    return float(np.max(projected) - np.min(projected))


def _catalog_lookup(shock_catalog, key, nearest, near, fill_value):
    out = np.full(near.shape[0], fill_value)
    if key in shock_catalog:
        out[near] = np.asarray(shock_catalog[key])[nearest[near]]
    return out


def _classification_length(classification):
    for key in ("near_shock", "crossed"):
        if key in classification:
            return np.asarray(classification[key]).shape[0]
    for value in classification.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            return arr.shape[0]
    raise ValueError("classification does not contain any array-like results")


def _classification_keep_mask(classification, keep, n_galaxies):
    if keep is None or keep == "all":
        return np.ones(n_galaxies, dtype=bool)
    if keep == "near_or_crossed":
        near = np.asarray(classification.get("near_shock", False), dtype=bool)
        crossed = np.asarray(classification.get("crossed", False), dtype=bool)
        return near | crossed
    if keep == "near_shock":
        return np.asarray(classification["near_shock"], dtype=bool)
    if keep == "crossed":
        return np.asarray(classification["crossed"], dtype=bool)

    keep_mask = np.asarray(keep, dtype=bool)
    if keep_mask.shape != (n_galaxies,):
        raise ValueError("custom keep mask must have shape (n_galaxies,)")
    return keep_mask


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
        "nearest_component_id": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_size": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_extent": np.full(n_galaxies, np.nan, dtype=np.float64),
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
    galaxy_pos_prev = ...   # The galaxy positions at the previous snapshot, shape (ngal, 3) in km.
    galaxy_pos_now = ...    # The galaxy positions at the current snapshot, shape (ngal, 3) in km.

    result, dissipation = run_shockfinder(cell)
    catalog = shock_front_catalog(result, dissipation, min_mach=1.5)
    catalog = filter_large_shock_fronts(
        catalog,
        link_length_km=50.0 * KPC_IN_KM,
        min_cells=30,
        min_extent_km=300.0 * KPC_IN_KM,
    )
    classification = classify_galaxy_shock_crossing(
        galaxy_pos_prev,
        galaxy_pos_now,
        catalog,
        search_radius_km=100.0 * KPC_IN_KM,
    )

    print("N galaxies near shocks:", np.count_nonzero(classification["near_shock"]))
    print("N galaxies crossed shocks:", np.count_nonzero(classification["crossed"]))

    classification = finalize_shock_classification(
        classification,
        result=result,
        dissipation=dissipation,
        shock_catalog=catalog,
        keep="near_or_crossed",
    )
