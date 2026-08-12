from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .catalog import ShockCatalog, ShockGroup


CATALOG_SCHEMA_VERSION = 1


def save_shock_catalog(path, catalog: ShockCatalog, *, compressed: bool = True) -> Path:
    """Save a complete catalog as a non-pickle NPZ archive."""

    output = Path(path)
    arrays = _catalog_arrays(catalog)
    writer = np.savez_compressed if compressed else np.savez
    with output.open("wb") as stream:
        writer(stream, **arrays)
    return output


def load_shock_catalog(path) -> ShockCatalog:
    """Load a catalog written by :func:`save_shock_catalog`."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = _required_fields()
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"catalog archive is missing fields: {', '.join(missing)}")
        version = int(np.asarray(archive["schema_version"]).item())
        if version != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported catalog schema version {version}; "
                f"expected {CATALOG_SCHEMA_VERSION}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in required}

    group_count = arrays["group_ids"].size
    _validate_group_shapes(arrays, group_count)
    offsets = arrays["center_offsets"]
    if offsets.shape != (group_count + 1,) or offsets[0] != 0:
        raise ValueError("catalog center offsets are invalid")
    if np.any(np.diff(offsets) < 0) or offsets[-1] != arrays["center_indices"].size:
        raise ValueError("catalog center offsets do not match center indices")
    if not np.array_equal(np.diff(offsets), arrays["n_centers"]):
        raise ValueError("catalog center counts do not match center offsets")

    groups: list[ShockGroup] = []
    for index in range(group_count):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        centers = arrays["center_indices"][start:stop].astype(np.int64, copy=True)
        groups.append(
            ShockGroup(
                group_id=int(arrays["group_ids"][index]),
                center_indices=centers,
                n_centers=int(arrays["n_centers"][index]),
                mach_peak=float(arrays["mach_peak"][index]),
                mach_mean=float(arrays["mach_mean"][index]),
                area=float(arrays["area"][index]),
                area_unit=str(arrays["area_unit"][index]),
                dissipation_total=float(arrays["dissipation_total"][index]),
                centroid=arrays["centroid"][index].astype(np.float64, copy=True),
                mean_normal=arrays["mean_normal"][index].astype(np.float64, copy=True),
                bounds=arrays["bounds"][index].astype(np.float64, copy=True),
                level_min=int(arrays["level_min"][index]),
                level_max=int(arrays["level_max"][index]),
                upstream_temperature=float(arrays["upstream_temperature"][index]),
                upstream_density=float(arrays["upstream_density"][index]),
                external_fraction=float(arrays["external_fraction"][index]),
                classification=str(arrays["classification"][index]),
            )
        )
    return ShockCatalog(
        group_id=arrays["cell_group_id"].astype(np.int64, copy=False),
        center_representative=arrays["center_representative"].astype(np.int64, copy=False),
        groups=groups,
    )


def save_shock_catalog_csv(path, catalog: ShockCatalog) -> Path:
    """Write one human-readable summary row per shock group."""

    output = Path(path)
    fields = [
        "group_id", "n_centers", "mach_peak", "mach_mean", "area",
        "area_unit", "dissipation_total", "centroid_x", "centroid_y",
        "centroid_z", "normal_x", "normal_y", "normal_z", "xmin", "xmax",
        "ymin", "ymax", "zmin", "zmax", "level_min", "level_max",
        "upstream_temperature", "upstream_density", "external_fraction",
        "classification",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in catalog.groups:
            writer.writerow(_group_csv_row(group))
    return output


def _catalog_arrays(catalog: ShockCatalog) -> dict[str, np.ndarray]:
    groups = catalog.groups
    centers = [np.asarray(group.center_indices, dtype=np.int64) for group in groups]
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([value.size for value in centers])
        center_indices = np.concatenate(centers)
    else:
        center_indices = np.empty(0, dtype=np.int64)

    def values(name, dtype):
        return np.asarray([getattr(group, name) for group in groups], dtype=dtype)

    return {
        "schema_version": np.asarray(CATALOG_SCHEMA_VERSION, dtype=np.int32),
        "cell_group_id": np.asarray(catalog.group_id, dtype=np.int64),
        "center_representative": np.asarray(catalog.center_representative, dtype=np.int64),
        "group_ids": values("group_id", np.int64),
        "center_offsets": offsets,
        "center_indices": center_indices,
        "n_centers": values("n_centers", np.int64),
        "mach_peak": values("mach_peak", np.float64),
        "mach_mean": values("mach_mean", np.float64),
        "area": values("area", np.float64),
        "area_unit": values("area_unit", "U32"),
        "dissipation_total": values("dissipation_total", np.float64),
        "centroid": _vectors(groups, "centroid", (3,)),
        "mean_normal": _vectors(groups, "mean_normal", (3,)),
        "bounds": _vectors(groups, "bounds", (2, 3)),
        "level_min": values("level_min", np.int32),
        "level_max": values("level_max", np.int32),
        "upstream_temperature": values("upstream_temperature", np.float64),
        "upstream_density": values("upstream_density", np.float64),
        "external_fraction": values("external_fraction", np.float64),
        "classification": values("classification", "U32"),
    }


def _vectors(groups, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not groups:
        return np.empty((0,) + shape, dtype=np.float64)
    return np.stack([np.asarray(getattr(group, name), dtype=np.float64) for group in groups])


def _required_fields() -> set[str]:
    return {
        "schema_version", "cell_group_id", "center_representative", "group_ids",
        "center_offsets", "center_indices", "n_centers", "mach_peak", "mach_mean",
        "area", "area_unit", "dissipation_total", "centroid", "mean_normal",
        "bounds", "level_min", "level_max", "upstream_temperature",
        "upstream_density", "external_fraction", "classification",
    }


def _validate_group_shapes(arrays: dict[str, np.ndarray], count: int) -> None:
    vector_shapes = {
        "centroid": (count, 3),
        "mean_normal": (count, 3),
        "bounds": (count, 2, 3),
    }
    for name, expected in vector_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"catalog field {name} has invalid shape")
    excluded = {
        "schema_version", "cell_group_id", "center_representative",
        "center_offsets", "center_indices", *vector_shapes,
    }
    for name, values in arrays.items():
        if name not in excluded and values.shape != (count,):
            raise ValueError(f"catalog field {name} has invalid shape")


def _group_csv_row(group: ShockGroup) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "n_centers": group.n_centers,
        "mach_peak": group.mach_peak,
        "mach_mean": group.mach_mean,
        "area": group.area,
        "area_unit": group.area_unit,
        "dissipation_total": group.dissipation_total,
        "centroid_x": group.centroid[0], "centroid_y": group.centroid[1],
        "centroid_z": group.centroid[2], "normal_x": group.mean_normal[0],
        "normal_y": group.mean_normal[1], "normal_z": group.mean_normal[2],
        "xmin": group.bounds[0, 0], "xmax": group.bounds[1, 0],
        "ymin": group.bounds[0, 1], "ymax": group.bounds[1, 1],
        "zmin": group.bounds[0, 2], "zmax": group.bounds[1, 2],
        "level_min": group.level_min, "level_max": group.level_max,
        "upstream_temperature": group.upstream_temperature,
        "upstream_density": group.upstream_density,
        "external_fraction": group.external_fraction,
        "classification": group.classification,
    }
