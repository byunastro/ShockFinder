from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .core import ShockFinder, ShockResult


@dataclass(slots=True)
class ShockGroup:
    """Summary of one face-connected physical shock surface."""

    group_id: int
    center_indices: np.ndarray
    n_centers: int
    mach_peak: float
    mach_mean: float
    area: float
    area_unit: str
    dissipation_total: float
    centroid: np.ndarray
    mean_normal: np.ndarray
    bounds: np.ndarray
    level_min: int
    level_max: int
    upstream_temperature: float
    upstream_density: float
    external_fraction: float
    classification: str
    boundary_faces: np.ndarray
    touches_boundary: bool
    valid_upstream_fraction: float
    classification_confidence: float
    mach_std: float
    normal_dispersion: float
    zone_width_mean: float
    zone_width_min: float
    zone_width_max: float
    level_count: int
    is_complete: bool
    quality_flags: tuple[str, ...]


@dataclass(slots=True)
class ShockCatalog:
    """Per-cell group labels and per-surface summaries."""

    group_id: np.ndarray
    center_representative: np.ndarray
    groups: list[ShockGroup]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CatalogSensitivity:
    """One summary row from a shock-catalog parameter sweep."""

    mach_tolerance: float
    normal_cosine: float
    duplicate_normal_cosine: float
    min_mach: float
    n_input_centers: int
    n_representative_centers: int
    n_groups: int
    surface_area: float
    dissipation_total: float
    mach_peak: float
    mach_mean: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != root:
            following = int(self.parent[value])
            self.parent[value] = root
            value = following
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_shock_catalog(
    result: ShockResult,
    *,
    cell=None,
    levels=None,
    dissipation=None,
    mach_tolerance: float = 0.3,
    normal_cosine: float = 0.7,
    deduplicate: bool = False,
    duplicate_normal_cosine: float = 0.8,
    min_mach: float = 1.0,
    boundary: str = "open",
    external_temperature: float = 1.0e4,
    classification_fraction: float = 0.8,
    boundary_margin_cells: float = 0.0,
    minimum_group_centers: int = 2,
    maximum_normal_dispersion: float = 0.3,
    provenance: dict[str, object] | None = None,
    _neighbor_tables=None,
) -> ShockCatalog:
    """Group AMR face-connected shock centers and summarize each surface.

    ``mach_tolerance`` is the maximum relative Mach difference between linked
    centers. ``normal_cosine`` uses the absolute dot product, so normals with
    opposite signs but the same surface orientation may still be connected.
    """

    if result.pos is None or result.dx is None:
        raise ValueError("result.pos and result.dx are required for shock grouping")
    if boundary != "open":
        raise ValueError(
            "boundary must be 'open'; extracted snapshot regions must not wrap"
        )
    if mach_tolerance < 0.0:
        raise ValueError("mach_tolerance must be non-negative")
    if not 0.0 <= normal_cosine <= 1.0:
        raise ValueError("normal_cosine must be between 0 and 1")
    if not 0.0 <= duplicate_normal_cosine <= 1.0:
        raise ValueError("duplicate_normal_cosine must be between 0 and 1")
    if min_mach < 1.0:
        raise ValueError("min_mach must be at least 1.0")
    if not np.isfinite(external_temperature) or external_temperature <= 0.0:
        raise ValueError("external_temperature must be finite and positive")
    if not 0.5 <= classification_fraction <= 1.0:
        raise ValueError("classification_fraction must be between 0.5 and 1.0")
    if boundary_margin_cells < 0.0:
        raise ValueError("boundary_margin_cells must be non-negative")
    if minimum_group_centers < 1:
        raise ValueError("minimum_group_centers must be at least 1")
    if not 0.0 <= maximum_normal_dispersion <= 1.0:
        raise ValueError("maximum_normal_dispersion must be between 0 and 1")

    n = result.mach.size
    if result.pos.shape != (n, 3) or result.dx.shape != (n,):
        raise ValueError("result geometry must have one row per retained cell")

    if levels is None:
        if result.level is not None:
            levels_array = np.asarray(result.level, dtype=np.int32)
        elif n == 0:
            levels_array = np.empty(0, dtype=np.int32)
        else:
            inferred = np.rint(np.log2(np.max(result.dx) / result.dx))
            levels_array = inferred.astype(np.int32)
    else:
        levels_array = np.asarray(levels, dtype=np.int32)
    if levels_array.shape != (n,):
        raise ValueError("levels must have one value per retained cell")

    shock_rows = np.nonzero(result.shock & (result.mach >= min_mach))[0]
    labels = np.full(n, -1, dtype=np.int64)
    representatives = np.full(n, -1, dtype=np.int64)
    if shock_rows.size == 0:
        return ShockCatalog(
            group_id=labels,
            center_representative=representatives,
            groups=[],
            metadata=_catalog_metadata(
                result,
                mach_tolerance,
                normal_cosine,
                deduplicate,
                duplicate_normal_cosine,
                min_mach,
                external_temperature,
                classification_fraction,
                provenance,
            ),
        )

    if _neighbor_tables is None:
        neighbors, fine_neighbors = ShockFinder._build_neighbor_tables(
            np.asfortranarray(result.pos, dtype=np.float64),
            np.asfortranarray(result.dx, dtype=np.float64),
            np.asfortranarray(levels_array, dtype=np.int32),
        )
    else:
        neighbors, fine_neighbors = _neighbor_tables
    shock_position = np.full(n, -1, dtype=np.int64)
    shock_position[shock_rows] = np.arange(shock_rows.size)
    union_find = _UnionFind(shock_rows.size)

    normals = result.normal
    if normals is None:
        normals = np.zeros((n, 3), dtype=np.float64)

    representatives[shock_rows] = shock_rows
    if deduplicate:
        representatives = _center_representatives(
            result,
            shock_rows,
            neighbors,
            fine_neighbors,
            normals,
            duplicate_normal_cosine,
        )
        shock_rows = shock_rows[representatives[shock_rows] == shock_rows]
        shock_position.fill(-1)
        shock_position[shock_rows] = np.arange(shock_rows.size)
        union_find = _UnionFind(shock_rows.size)

    for local_i, row in enumerate(shock_rows):
        adjacent = np.concatenate((neighbors[row], fine_neighbors[row].ravel()))
        for encoded in adjacent:
            if encoded <= 0:
                continue
            other = int(encoded) - 1
            local_j = int(shock_position[other])
            if local_j < 0:
                continue
            scale = max(float(result.mach[row]), float(result.mach[other]))
            if abs(float(result.mach[row]) - float(result.mach[other])) / scale > mach_tolerance:
                continue
            ni = normals[row]
            nj = normals[other]
            ni_norm = float(np.linalg.norm(ni))
            nj_norm = float(np.linalg.norm(nj))
            if ni_norm > 0.0 and nj_norm > 0.0:
                alignment = abs(float(np.dot(ni, nj))) / (ni_norm * nj_norm)
                if alignment < normal_cosine:
                    continue
            union_find.union(local_i, local_j)

    roots = np.array([union_find.find(i) for i in range(shock_rows.size)], dtype=np.int64)
    unique_roots = np.unique(roots)
    for group_id, root in enumerate(unique_roots):
        labels[shock_rows[roots == root]] = group_id

    area = _surface_area(result, shock_rows)
    total = np.zeros(n, dtype=np.float64)
    if dissipation is not None:
        area = np.asarray(dissipation.area, dtype=np.float64)
        total = np.asarray(dissipation.total, dtype=np.float64)
        if area.shape != (n,) or total.shape != (n,):
            raise ValueError("dissipation arrays must have one value per retained cell")

    upstream_temperature, upstream_density = _upstream_fields(cell, result)
    region_lower = np.min(result.pos - 0.5 * result.dx[:, None], axis=0)
    region_upper = np.max(result.pos + 0.5 * result.dx[:, None], axis=0)

    groups: list[ShockGroup] = []
    for group_id in range(unique_roots.size):
        rows = np.nonzero(labels == group_id)[0]
        weights = area[rows]
        if not np.any(weights > 0.0):
            weights = np.ones(rows.size, dtype=np.float64)
        weight_sum = float(np.sum(weights))
        centroid = np.average(result.pos[rows], axis=0, weights=weights)
        mean_normal = np.average(normals[rows], axis=0, weights=weights)
        normal_length = float(np.linalg.norm(mean_normal))
        if normal_length > 0.0:
            mean_normal /= normal_length
        half_width = 0.5 * result.dx[rows, None]
        lower = np.min(result.pos[rows] - half_width, axis=0)
        upper = np.max(result.pos[rows] + half_width, axis=0)
        valid_upstream = np.isfinite(upstream_temperature[rows])
        if np.any(valid_upstream):
            upstream_weights = weights[valid_upstream]
            upstream_weight_sum = float(np.sum(upstream_weights))
            group_temperature = float(
                np.sum(upstream_temperature[rows][valid_upstream] * upstream_weights)
                / upstream_weight_sum
            )
            group_density = float(
                np.sum(upstream_density[rows][valid_upstream] * upstream_weights)
                / upstream_weight_sum
            )
            external_fraction = float(
                np.sum(
                    upstream_weights[
                        upstream_temperature[rows][valid_upstream]
                        <= external_temperature
                    ]
                )
                / upstream_weight_sum
            )
            valid_upstream_fraction = float(
                np.sum(upstream_weights) / weight_sum
            )
            if external_fraction >= classification_fraction:
                classification = "external"
            elif external_fraction <= 1.0 - classification_fraction:
                classification = "internal"
            else:
                classification = "mixed"
        else:
            group_temperature = np.nan
            group_density = np.nan
            external_fraction = np.nan
            classification = "unclassified"
            valid_upstream_fraction = 0.0

        mach_mean = float(np.sum(result.mach[rows] * weights) / weight_sum)
        mach_std = float(
            np.sqrt(np.sum((result.mach[rows] - mach_mean) ** 2 * weights) / weight_sum)
        )
        group_normals = normals[rows].copy()
        valid_normals = np.linalg.norm(group_normals, axis=1) > 0.0
        if np.any(valid_normals):
            reference = group_normals[np.nonzero(valid_normals)[0][0]]
            flips = np.sum(group_normals * reference, axis=1) < 0.0
            group_normals[flips] *= -1.0
            normal_resultant = np.average(group_normals[valid_normals], axis=0, weights=weights[valid_normals])
            normal_dispersion = float(1.0 - np.clip(np.linalg.norm(normal_resultant), 0.0, 1.0))
        else:
            normal_dispersion = 1.0
        zone_values = (
            np.asarray(result.zone_width[rows], dtype=np.float64)
            if result.zone_width is not None
            else np.zeros(rows.size, dtype=np.float64)
        )
        positive_zone = zone_values > 0.0
        if np.any(positive_zone):
            zone_width_mean = float(np.average(zone_values[positive_zone], weights=weights[positive_zone]))
            zone_width_min = float(np.min(zone_values[positive_zone]))
            zone_width_max = float(np.max(zone_values[positive_zone]))
        else:
            zone_width_mean = zone_width_min = zone_width_max = 0.0
        margin = boundary_margin_cells * result.dx[rows, None]
        cell_lower = result.pos[rows] - 0.5 * result.dx[rows, None]
        cell_upper = result.pos[rows] + 0.5 * result.dx[rows, None]
        boundary_faces = np.array(
            [
                np.any(cell_lower[:, 0] - margin[:, 0] <= region_lower[0]),
                np.any(cell_upper[:, 0] + margin[:, 0] >= region_upper[0]),
                np.any(cell_lower[:, 1] - margin[:, 0] <= region_lower[1]),
                np.any(cell_upper[:, 1] + margin[:, 0] >= region_upper[1]),
                np.any(cell_lower[:, 2] - margin[:, 0] <= region_lower[2]),
                np.any(cell_upper[:, 2] + margin[:, 0] >= region_upper[2]),
            ],
            dtype=bool,
        )
        touches_boundary = bool(np.any(boundary_faces))
        classification_confidence = (
            float(max(external_fraction, 1.0 - external_fraction))
            if np.isfinite(external_fraction)
            else np.nan
        )
        flags: list[str] = []
        if touches_boundary:
            flags.append("touches_boundary")
        if valid_upstream_fraction < 1.0:
            flags.append("missing_upstream")
        if rows.size < minimum_group_centers:
            flags.append("small_group")
        if normal_dispersion > maximum_normal_dispersion:
            flags.append("normal_dispersion")
        is_complete = not touches_boundary and valid_upstream_fraction == 1.0
        groups.append(
            ShockGroup(
                group_id=group_id,
                center_indices=rows,
                n_centers=rows.size,
                mach_peak=float(np.max(result.mach[rows])),
                mach_mean=mach_mean,
                area=weight_sum,
                area_unit="kpc2" if dissipation is not None else "position_unit2",
                dissipation_total=float(np.sum(total[rows])),
                centroid=np.asarray(centroid),
                mean_normal=np.asarray(mean_normal),
                bounds=np.stack((lower, upper)),
                level_min=int(np.min(levels_array[rows])),
                level_max=int(np.max(levels_array[rows])),
                upstream_temperature=group_temperature,
                upstream_density=group_density,
                external_fraction=external_fraction,
                classification=classification,
                boundary_faces=boundary_faces,
                touches_boundary=touches_boundary,
                valid_upstream_fraction=valid_upstream_fraction,
                classification_confidence=classification_confidence,
                mach_std=mach_std,
                normal_dispersion=normal_dispersion,
                zone_width_mean=zone_width_mean,
                zone_width_min=zone_width_min,
                zone_width_max=zone_width_max,
                level_count=int(np.unique(levels_array[rows]).size),
                is_complete=is_complete,
                quality_flags=tuple(flags),
            )
        )
    groups = sorted(
        groups,
        key=lambda group: (
            -group.mach_peak,
            *np.asarray(group.centroid, dtype=np.float64).tolist(),
            -group.area,
        ),
    )
    old_to_new = np.full(len(groups), -1, dtype=np.int64)
    for new_id, group in enumerate(groups):
        old_to_new[group.group_id] = new_id
        group.group_id = new_id
    grouped = labels >= 0
    labels[grouped] = old_to_new[labels[grouped]]
    return ShockCatalog(
        group_id=labels,
        center_representative=representatives,
        groups=groups,
        metadata=_catalog_metadata(
            result,
            mach_tolerance,
            normal_cosine,
            deduplicate,
            duplicate_normal_cosine,
            min_mach,
            external_temperature,
            classification_fraction,
            provenance,
        ),
    )


def _catalog_metadata(
    result,
    mach_tolerance,
    normal_cosine,
    deduplicate,
    duplicate_normal_cosine,
    min_mach,
    external_temperature,
    classification_fraction,
    provenance,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "retained_cells": int(result.mach.size),
        "mach_tolerance": float(mach_tolerance),
        "normal_cosine": float(normal_cosine),
        "deduplicate": bool(deduplicate),
        "duplicate_normal_cosine": float(duplicate_normal_cosine),
        "min_mach": float(min_mach),
        "external_temperature": float(external_temperature),
        "classification_fraction": float(classification_fraction),
        "boundary": "open",
    }
    if result.level is not None and result.level.size:
        metadata["level_min"] = int(np.min(result.level))
        metadata["level_max"] = int(np.max(result.level))
    if provenance:
        metadata.update(provenance)
    return metadata


def _upstream_fields(cell, result: ShockResult) -> tuple[np.ndarray, np.ndarray]:
    n = result.mach.size
    temperature = np.full(n, np.nan, dtype=np.float64)
    density = np.full(n, np.nan, dtype=np.float64)
    if cell is None:
        return temperature, density

    temp_all = ShockFinder._field(
        cell,
        (("T", "K"), ("temperature", "K"), "T", "temp", "temperature"),
    )
    rho_all = ShockFinder._field(
        cell,
        (("rho", "Msol/kpc3"), "rho", "density"),
    )
    valid = (result.upstream_index >= 0) & (result.upstream_index < n)
    if not np.any(valid):
        return temperature, density
    upstream_retained = result.upstream_index[valid]
    upstream_original = result.selected_indices[upstream_retained]
    temperature[valid] = np.asarray(temp_all, dtype=np.float64)[upstream_original]
    density[valid] = np.asarray(rho_all, dtype=np.float64)[upstream_original]
    return temperature, density


def analyze_catalog_sensitivity(
    result: ShockResult,
    *,
    dissipation=None,
    mach_tolerances=(0.2, 0.3, 0.5),
    normal_cosines=(0.5, 0.7, 0.9),
    duplicate_normal_cosines=(0.8,),
    min_machs=(1.3,),
    deduplicate: bool = True,
    boundary: str = "open",
) -> list[CatalogSensitivity]:
    """Evaluate catalog stability across grouping and selection thresholds."""

    rows: list[CatalogSensitivity] = []
    n = result.mach.size
    if result.pos is None or result.dx is None:
        raise ValueError("result.pos and result.dx are required for sensitivity analysis")
    if result.level is not None:
        levels = np.asarray(result.level, dtype=np.int32)
    elif n == 0:
        levels = np.empty(0, dtype=np.int32)
    else:
        levels = np.rint(np.log2(np.max(result.dx) / result.dx)).astype(np.int32)
    neighbor_tables = ShockFinder._build_neighbor_tables(
        np.asfortranarray(result.pos, dtype=np.float64),
        np.asfortranarray(result.dx, dtype=np.float64),
        np.asfortranarray(levels, dtype=np.int32),
    )
    for min_mach in min_machs:
        for mach_tolerance in mach_tolerances:
            for normal_cosine in normal_cosines:
                for duplicate_normal_cosine in duplicate_normal_cosines:
                    catalog = build_shock_catalog(
                        result,
                        dissipation=dissipation,
                        mach_tolerance=float(mach_tolerance),
                        normal_cosine=float(normal_cosine),
                        deduplicate=deduplicate,
                        duplicate_normal_cosine=float(duplicate_normal_cosine),
                        min_mach=float(min_mach),
                        boundary=boundary,
                        levels=levels,
                        _neighbor_tables=neighbor_tables,
                    )
                    selected = result.shock & (result.mach >= float(min_mach))
                    representatives = catalog.center_representative
                    representative_count = int(
                        np.count_nonzero(selected & (representatives == np.arange(result.mach.size)))
                    )
                    areas = np.array([group.area for group in catalog.groups], dtype=np.float64)
                    group_mach = np.array([group.mach_mean for group in catalog.groups], dtype=np.float64)
                    area_sum = float(np.sum(areas))
                    mean_mach = (
                        float(np.sum(group_mach * areas) / area_sum)
                        if area_sum > 0.0
                        else 0.0
                    )
                    rows.append(
                        CatalogSensitivity(
                            mach_tolerance=float(mach_tolerance),
                            normal_cosine=float(normal_cosine),
                            duplicate_normal_cosine=float(duplicate_normal_cosine),
                            min_mach=float(min_mach),
                            n_input_centers=int(np.count_nonzero(selected)),
                            n_representative_centers=representative_count,
                            n_groups=len(catalog.groups),
                            surface_area=area_sum,
                            dissipation_total=float(
                                sum(group.dissipation_total for group in catalog.groups)
                            ),
                            mach_peak=float(
                                max((group.mach_peak for group in catalog.groups), default=0.0)
                            ),
                            mach_mean=mean_mach,
                        )
                    )
    return rows


def _center_representatives(
    result: ShockResult,
    shock_rows: np.ndarray,
    neighbors: np.ndarray,
    fine_neighbors: np.ndarray,
    normals: np.ndarray,
    normal_cosine: float,
) -> np.ndarray:
    """Return the strongest representative for centers stacked along a normal."""

    n = result.mach.size
    representatives = np.full(n, -1, dtype=np.int64)
    representatives[shock_rows] = shock_rows
    shock_position = np.full(n, -1, dtype=np.int64)
    shock_position[shock_rows] = np.arange(shock_rows.size)
    union_find = _UnionFind(shock_rows.size)

    for local_i, row in enumerate(shock_rows):
        ni = normals[row]
        ni_norm = float(np.linalg.norm(ni))
        if ni_norm == 0.0:
            continue
        adjacent = np.concatenate((neighbors[row], fine_neighbors[row].ravel()))
        for encoded in adjacent:
            if encoded <= 0:
                continue
            other = int(encoded) - 1
            local_j = int(shock_position[other])
            if local_j < 0:
                continue
            displacement = result.pos[other] - result.pos[row]
            distance = float(np.linalg.norm(displacement))
            if distance == 0.0:
                continue
            direction = displacement / distance
            nj = normals[other]
            nj_norm = float(np.linalg.norm(nj))
            if nj_norm == 0.0:
                continue
            normal_alignment = abs(float(np.dot(ni, nj))) / (ni_norm * nj_norm)
            along_i = abs(float(np.dot(direction, ni))) / ni_norm
            along_j = abs(float(np.dot(direction, nj))) / nj_norm
            if min(normal_alignment, along_i, along_j) >= normal_cosine:
                union_find.union(local_i, local_j)

    members: dict[int, list[int]] = {}
    for local_i, row in enumerate(shock_rows):
        members.setdefault(union_find.find(local_i), []).append(int(row))
    for rows in members.values():
        representative = min(rows, key=lambda row: (-float(result.mach[row]), row))
        representatives[rows] = representative
    return representatives


def _surface_area(result: ShockResult, rows: np.ndarray) -> np.ndarray:
    base_area = result.dx**2
    area = base_area.copy()
    if result.normal is None:
        return area
    dominant = np.max(np.abs(result.normal[rows]), axis=1)
    valid = dominant > 0.0
    corrected = base_area[rows].copy()
    corrected[valid] /= np.clip(dominant[valid], 1.0 / np.sqrt(3.0), 1.0)
    area[rows] = corrected
    return area
