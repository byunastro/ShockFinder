from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(slots=True)
class ShockCatalog:
    """Per-cell group labels and per-surface summaries."""

    group_id: np.ndarray
    center_representative: np.ndarray
    groups: list[ShockGroup]


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
    levels=None,
    dissipation=None,
    mach_tolerance: float = 0.3,
    normal_cosine: float = 0.7,
    deduplicate: bool = False,
    duplicate_normal_cosine: float = 0.8,
) -> ShockCatalog:
    """Group AMR face-connected shock centers and summarize each surface.

    ``mach_tolerance`` is the maximum relative Mach difference between linked
    centers. ``normal_cosine`` uses the absolute dot product, so normals with
    opposite signs but the same surface orientation may still be connected.
    """

    if result.pos is None or result.dx is None:
        raise ValueError("result.pos and result.dx are required for shock grouping")
    if mach_tolerance < 0.0:
        raise ValueError("mach_tolerance must be non-negative")
    if not 0.0 <= normal_cosine <= 1.0:
        raise ValueError("normal_cosine must be between 0 and 1")
    if not 0.0 <= duplicate_normal_cosine <= 1.0:
        raise ValueError("duplicate_normal_cosine must be between 0 and 1")

    n = result.mach.size
    if result.pos.shape != (n, 3) or result.dx.shape != (n,):
        raise ValueError("result geometry must have one row per retained cell")

    if levels is None:
        if result.level is not None:
            levels_array = np.asarray(result.level, dtype=np.int32)
        else:
            inferred = np.rint(np.log2(np.max(result.dx) / result.dx))
            levels_array = inferred.astype(np.int32)
    else:
        levels_array = np.asarray(levels, dtype=np.int32)
    if levels_array.shape != (n,):
        raise ValueError("levels must have one value per retained cell")

    shock_rows = np.nonzero(result.shock & (result.mach > 1.0))[0]
    labels = np.full(n, -1, dtype=np.int64)
    representatives = np.full(n, -1, dtype=np.int64)
    if shock_rows.size == 0:
        return ShockCatalog(group_id=labels, center_representative=representatives, groups=[])

    neighbors, fine_neighbors = ShockFinder._build_neighbor_tables(
        np.asfortranarray(result.pos, dtype=np.float64),
        np.asfortranarray(result.dx, dtype=np.float64),
        np.asfortranarray(levels_array, dtype=np.int32),
    )
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
        groups.append(
            ShockGroup(
                group_id=group_id,
                center_indices=rows,
                n_centers=rows.size,
                mach_peak=float(np.max(result.mach[rows])),
                mach_mean=float(np.sum(result.mach[rows] * weights) / weight_sum),
                area=weight_sum,
                area_unit="kpc2" if dissipation is not None else "position_unit2",
                dissipation_total=float(np.sum(total[rows])),
                centroid=np.asarray(centroid),
                mean_normal=np.asarray(mean_normal),
                bounds=np.stack((lower, upper)),
                level_min=int(np.min(levels_array[rows])),
                level_max=int(np.max(levels_array[rows])),
            )
        )
    return ShockCatalog(
        group_id=labels,
        center_representative=representatives,
        groups=groups,
    )


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
