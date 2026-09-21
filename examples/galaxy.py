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
from shocktest.spatial import nearest_points, connected_components, nearest_segment_shock


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
    shock_zone_width = (
        np.asarray(result.zone_width, dtype=np.float64)[shock_rows]
        if getattr(result, "zone_width", None) is not None
        else np.zeros(shock_rows.size, dtype=np.float64)
    )

    upstream = result.upstream_index[shock_rows]
    downstream = result.downstream_index[shock_rows]
    valid = (upstream >= 0) & (downstream >= 0)

    normal = np.zeros_like(shock_pos)
    normal[valid] = (result.normal[shock_rows[valid]] if result.normal is not None
                     else result.pos[downstream[valid]] - result.pos[upstream[valid]])
    norm = np.linalg.norm(normal, axis=1)
    valid &= norm > 0.0
    normal[valid] /= norm[valid, None]

    return {
        "rows": shock_rows,
        "pos": shock_pos,
        "dx": shock_dx,
        "mach": shock_mach,
        "flux": shock_flux,
        "zone_width": shock_zone_width,
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
    zone_width_factor=0.5,
    memory_budget_bytes=32 * 1024**2,
):
    """Classify whether each galaxy crossed or entered a nearby shock zone.

    Parameters
    ----------
    galaxy_pos_prev, galaxy_pos_now:
        Galaxy positions at two snapshots, in km, with matching row order.
    shock_catalog:
        Output from ``shock_front_catalog``.
    search_radius_km:
        Maximum distance from the galaxy trajectory samples to a shock cell.
    width_factor:
        Geometric tolerance in units of the local shock-cell ``dx``.
    zone_width_factor:
        Multiplier applied to ``shock_catalog["zone_width"]`` when building the
        finite shock-zone half-width. The default 0.5 treats ``zone_width`` as
        the upstream-to-downstream full span around the shock center.
    """

    galaxy_pos_prev = np.asarray(galaxy_pos_prev, dtype=np.float64)
    galaxy_pos_now = np.asarray(galaxy_pos_now, dtype=np.float64)
    if galaxy_pos_now.ndim != 2 or galaxy_pos_prev.shape != galaxy_pos_now.shape or galaxy_pos_now.shape[1] != 3:
        raise ValueError("galaxy positions must both have shape (ngal, 3)")

    shock_pos = shock_catalog["pos"]
    shock_normal = shock_catalog["normal"]
    valid_normal = shock_catalog["valid_normal"]
    if shock_pos.size == 0:
        return _empty_classification(galaxy_pos_now.shape[0])

    nearest, distance = nearest_segment_shock(
        galaxy_pos_prev, galaxy_pos_now, shock_catalog,
        search_radius=search_radius_km, width_factor=width_factor,
        zone_width_factor=zone_width_factor, memory_budget_bytes=memory_budget_bytes,
    )
    catalog_zone_width = np.asarray(shock_catalog.get("zone_width", np.zeros(shock_pos.shape[0])), dtype=np.float64)
    search_zone_half_width = (
        zone_width_factor * catalog_zone_width[nearest]
        + width_factor * shock_catalog["dx"][nearest]
    )
    near = ((distance <= search_radius_km) | (distance <= search_zone_half_width)) & valid_normal[nearest]

    crossed = np.zeros(galaxy_pos_now.shape[0], dtype=bool)
    affected_zone = np.zeros(galaxy_pos_now.shape[0], dtype=bool)
    signed_prev = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    signed_now = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    zone_distance = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    zone_half_width = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)
    transverse = np.full(galaxy_pos_now.shape[0], np.nan, dtype=np.float64)

    if np.any(near):
        shock_idx = nearest[near]
        p0 = galaxy_pos_prev[near]
        p1 = galaxy_pos_now[near]
        xs = shock_pos[shock_idx]
        ns = shock_normal[shock_idx]

        signed_prev[near] = np.sum((p0 - xs) * ns, axis=1)
        signed_now[near] = np.sum((p1 - xs) * ns, axis=1)

        delta_signed = signed_now[near] - signed_prev[near]
        step = p1 - p0
        length2 = np.sum(step * step, axis=1)
        fraction = np.divide(np.sum((xs - p0) * step, axis=1), length2,
                             out=np.zeros(len(p0)), where=length2 > 0)
        fraction = np.divide(-signed_prev[near], delta_signed,
                             out=fraction, where=delta_signed != 0)
        intersection = p0 + np.clip(fraction, 0, 1)[:, None] * step
        offset = intersection - xs
        normal_offset = np.sum(offset * ns, axis=1)[:, None] * ns
        transverse[near] = np.linalg.norm(offset - normal_offset, axis=1)

        transverse_width = width_factor * shock_catalog["dx"][shock_idx]
        zone_half_width[near] = zone_width_factor * catalog_zone_width[shock_idx] + transverse_width
        zone_distance[near] = _minimum_segment_plane_distance(
            signed_prev[near],
            signed_now[near],
        )
        crossed[near] = (signed_prev[near] * signed_now[near] <= 0.0) & (transverse[near] <= transverse_width)
        affected_zone[near] = (zone_distance[near] <= zone_half_width[near]) & (transverse[near] <= transverse_width)

    return {
        "crossed": crossed,
        "affected_zone": affected_zone,
        "near_shock": near,
        "nearest_shock_row": np.where(near, shock_catalog["rows"][nearest], -1),
        "nearest_component_id": _catalog_lookup(shock_catalog, "component_id", nearest, near, -1),
        "nearest_component_size": _catalog_lookup(shock_catalog, "component_size", nearest, near, -1),
        "nearest_component_extent": _catalog_lookup(shock_catalog, "component_extent", nearest, near, np.nan),
        "nearest_mach": np.where(near, shock_catalog["mach"][nearest], np.nan),
        "nearest_flux": np.where(near, shock_catalog["flux"][nearest], np.nan),
        "nearest_zone_width": np.where(near, catalog_zone_width[nearest], np.nan),
        "distance_to_shock": np.where(near, distance, np.nan),
        "signed_distance_prev": signed_prev,
        "signed_distance_now": signed_now,
        "zone_distance": zone_distance,
        "zone_half_width": zone_half_width,
        "transverse_distance": transverse,
    }


def compact_classification_results(classification, *, keep="near_or_crossed", keep_keys=None):
    """Copy only useful galaxy-classification rows into a compact result dict.

    ``keep="near_or_crossed"`` keeps galaxies flagged by ``near_shock``,
    ``crossed``, or ``affected_zone`` and records their original row numbers as
    ``galaxy_index``.
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
    return connected_components(points, link_length)


def _connected_components_numpy(points, link_length, chunk_size=2048):
    return connected_components(points, link_length, use_scipy=False)


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
        affected = np.asarray(classification.get("affected_zone", False), dtype=bool)
        return near | crossed | affected
    if keep == "affected_zone":
        return np.asarray(classification["affected_zone"], dtype=bool)
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


def _nearest_shock_trajectory(points_prev, points_now, shock_pos):
    """Return nearest shock to the previous, current, or midpoint position."""

    points_mid = 0.5 * (points_prev + points_now)
    nearest_prev, distance_prev = _nearest_shock(points_prev, shock_pos)
    nearest_now, distance_now = _nearest_shock(points_now, shock_pos)
    nearest_mid, distance_mid = _nearest_shock(points_mid, shock_pos)

    nearest = nearest_now.copy()
    distance = distance_now.copy()

    use_prev = distance_prev < distance
    nearest[use_prev] = nearest_prev[use_prev]
    distance[use_prev] = distance_prev[use_prev]

    use_mid = distance_mid < distance
    nearest[use_mid] = nearest_mid[use_mid]
    distance[use_mid] = distance_mid[use_mid]
    return nearest, distance


def _minimum_segment_plane_distance(signed_prev, signed_now):
    crosses_plane = signed_prev * signed_now <= 0.0
    distance = np.minimum(np.abs(signed_prev), np.abs(signed_now))
    distance[crosses_plane] = 0.0
    return distance


def _nearest_shock_numpy(points, shock_pos, chunk_size=4096):
    return nearest_points(points, shock_pos)


def _empty_classification(n_galaxies):
    return {
        "crossed": np.zeros(n_galaxies, dtype=bool),
        "affected_zone": np.zeros(n_galaxies, dtype=bool),
        "near_shock": np.zeros(n_galaxies, dtype=bool),
        "nearest_shock_row": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_id": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_size": np.full(n_galaxies, -1, dtype=np.int64),
        "nearest_component_extent": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_mach": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_flux": np.full(n_galaxies, np.nan, dtype=np.float64),
        "nearest_zone_width": np.full(n_galaxies, np.nan, dtype=np.float64),
        "distance_to_shock": np.full(n_galaxies, np.nan, dtype=np.float64),
        "signed_distance_prev": np.full(n_galaxies, np.nan, dtype=np.float64),
        "signed_distance_now": np.full(n_galaxies, np.nan, dtype=np.float64),
        "zone_distance": np.full(n_galaxies, np.nan, dtype=np.float64),
        "zone_half_width": np.full(n_galaxies, np.nan, dtype=np.float64),
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
    print("N galaxies affected by shock zones:", np.count_nonzero(classification["affected_zone"]))

    classification = finalize_shock_classification(
        classification,
        result=result,
        dissipation=dissipation,
        shock_catalog=catalog,
        keep="near_or_crossed",
    )
