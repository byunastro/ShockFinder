"""Validated loading for the actual dense same-call ShockFinder producer.

The supplied producer passes its original result into compute_dissipation
with compact=False. A dense retained-row index is consequently an implicit
identifier in the dissipation file. At accepted centers we verify that the
saved center_index equals that retained-row ID before any keyed selection.
This contract, explicitly supplied by the user, is not inferred from lengths.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .inspect_inputs import KINDS, discover
from .model import KPC_KM, Metadata, Snapshot, node_key
from .pickle_metadata import array_metadata, object_fields, read_metadata


def read_metadata_table(path, selected, cfg):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"exact snapshot metadata are required: {path}; use shockTree.export_metadata on the simulation host")
    rows = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"timestep", "aexp", "time_gyr"} <= set(reader.fieldnames or ()):
            raise ValueError("metadata CSV requires timestep,aexp,time_gyr")
        for row in reader:
            t = int(row["timestep"])
            if t not in selected:
                continue
            if t in rows:
                raise ValueError(f"duplicate metadata row for snapshot {t}")
            a, age = float(row["aexp"]), float(row["time_gyr"])
            if not np.isfinite(a) or a <= 0 or not np.isfinite(age) or age < 0:
                raise ValueError(f"snapshot {t}: invalid scale factor/age")
            box = cfg.box_comoving_kpc
            if all(row.get(f"box_{axis}_comoving_kpc") for axis in "xyz"):
                table_box = [float(row[f"box_{axis}_comoving_kpc"]) for axis in "xyz"]
                if box is not None and cfg.periodic and not np.allclose(box, table_box, rtol=1e-8, atol=0):
                    raise ValueError(f"snapshot {t}: configured box and metadata box disagree")
                box = table_box if box is None else box
            if cfg.periodic:
                box = np.asarray(box, dtype=np.float64) if box is not None else None
                if box is None or box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
                    raise ValueError(f"snapshot {t}: periodic matching requires a positive three-dimensional comoving box")
            else:
                box = None
            rows[t] = Metadata(t, a, age, box)
    missing = set(selected) - rows.keys()
    if missing:
        raise ValueError(f"metadata missing for selected snapshots: {sorted(missing)}")
    ordered = [rows[t] for t in selected]
    if any(b.time_gyr <= a.time_gyr for a, b in zip(ordered, ordered[1:])):
        raise ValueError("physical times must increase along the sorted selected snapshots")
    if any(b.aexp < a.aexp for a, b in zip(ordered, ordered[1:])):
        raise ValueError("scale factors decrease in the sorted cosmological snapshot sequence")
    if cfg.periodic and any(not np.allclose(x.box_comoving_kpc, ordered[0].box_comoving_kpc, rtol=1e-8, atol=0)
                            for x in ordered[1:]):
        raise ValueError("periodic comoving box changes between snapshots")
    return rows


def selected_files(cfg):
    if cfg.manifest_path is None:
        found, excluded = discover(Path(cfg.input_dir), cfg.snapshot_start, cfg.snapshot_end)
    else:
        manifest = Path(cfg.manifest_path).resolve()
        found, excluded = {}, set()
        with manifest.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"timestep", "result_path", "dissipation_path", "catalog_path"} <= set(reader.fieldnames or ()):
                raise ValueError("manifest requires timestep,result_path,dissipation_path,catalog_path")
            for row in reader:
                t = int(row["timestep"])
                if not cfg.snapshot_start <= t <= cfg.snapshot_end:
                    excluded.add(t)
                    continue
                if t in found:
                    raise ValueError(f"manifest repeats snapshot {t}")
                found[t] = {}
                for kind in KINDS:
                    if not row[f"{kind}_path"]:
                        continue
                    path = Path(row[f"{kind}_path"])
                    found[t][kind] = path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()
        found, excluded = dict(sorted(found.items())), sorted(excluded)
        if not found:
            raise ValueError("manifest has no selected snapshots")
    for timestep, paths in found.items():
        missing = set(KINDS) - paths.keys()
        if missing:
            raise ValueError(f"snapshot {timestep}: missing required files {sorted(missing)}")
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"snapshot {timestep}: a required input path is not a file")
    return found, excluded


def _slice(path, metadata, lo, hi):
    """Copy one bounded row chunk and immediately close its payload mapping."""
    shape, dtype = metadata.shape, metadata.dtype
    if len(shape) == 1:
        if hi == lo:
            return np.empty(0, dtype=dtype)
        values = np.memmap(path, mode="r", dtype=dtype,
                           offset=metadata.offset + lo * dtype.itemsize, shape=(hi - lo,))
        output = values.copy()
        values._mmap.close()
        return output
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError("only scalar rows and three-dimensional vectors are supported")
    output = np.empty((hi - lo, 3), dtype=dtype)
    if hi == lo:
        return output
    if metadata.fortran_order:
        for axis in range(3):
            values = np.memmap(path, mode="r", dtype=dtype,
                               offset=metadata.offset + (axis * shape[0] + lo) * dtype.itemsize,
                               shape=(hi - lo,))
            output[:, axis] = values
            values._mmap.close()
    else:
        values = np.memmap(path, mode="r", dtype=dtype,
                           offset=metadata.offset + lo * 3 * dtype.itemsize, shape=(hi - lo, 3))
        output[:] = values
        values._mmap.close()
    return output


def _rows(path, metadata, rows):
    if not rows.size:
        return np.empty((0,) + metadata.shape[1:], dtype=metadata.dtype)
    lo, hi = int(rows.min()), int(rows.max()) + 1
    # Field-by-field bounds prevent all accessed pickle pages accumulating in RSS.
    return _slice(path, metadata, lo, hi)[rows - lo]


def _objects(paths, timestep):
    output = {}
    for kind in KINDS:
        object_type, fields = object_fields(read_metadata(paths[kind]))
        for name in ("timestep", "snapshot", "iout"):
            if name in fields and fields[name] != timestep:
                raise ValueError(f"snapshot {timestep}: {kind} embedded snapshot is inconsistent")
        output[kind] = (object_type, fields)
    return output


def load_snapshot(paths, meta: Metadata | None, cfg: PipelineConfig, invalid_sink=None, timestep=None):
    """Compact one snapshot; with meta=None, return raw input checks only.

    The caller retains only the two snapshots needed for a pair. Invalid
    records are reported and excluded because the minimal dtype has no valid
    flag and requires finite unit normals. Dissipation <=0 is a missing
    feature, not an invalid shock.
    """
    t = meta.timestep if meta is not None else timestep
    if t is None:
        raise ValueError("snapshot label is required")
    objects = _objects(paths, t)
    result_type, result = objects["result"]
    diss_type, diss = objects["dissipation"]
    if result_type not in {"shocktest.core.ShockResult", "builtins.dict"} or diss_type not in {"shocktest.pyShockFinder.DissipationResult", "builtins.dict"}:
        raise ValueError("unsupported result/dissipation object types")
    needed = {name: array_metadata(result[name]) for name in
              ("shock", "center_index", "mach", "pos", "normal", "dx")}
    n = needed["shock"].shape[0]
    for name, info in needed.items():
        if info.shape != ((n, 3) if name in {"pos", "normal"} else (n,)):
            raise ValueError(f"snapshot {t}: malformed result.{name}")
    if needed["shock"].dtype != np.dtype(bool) or needed["center_index"].dtype.kind not in "iu":
        raise ValueError("invalid shock mask or center ID dtype")
    diss_meta = {name: array_metadata(diss[name]) for name in ("flux", "total", "area", "efficiency", "sound_speed")}
    if any(value.shape != (n,) for value in diss_meta.values()):
        raise ValueError("dense producer contract violated: dissipation rows do not match retained-row IDs")
    catalog_type, catalog = objects["catalog"]
    if cfg.catalog_mode == "disabled_by_producer":
        if catalog_type != "builtins.NoneType":
            raise ValueError("catalog_mode disabled_by_producer requires the saved None catalog; use identified mode for records")
        catalog_ids = None
    else:
        if "shock_id" not in catalog:
            raise ValueError("identified catalog must explicitly supply shock_id")
        catmeta = array_metadata(catalog["shock_id"])
        if len(catmeta.shape) != 1 or catmeta.dtype.kind not in "iu":
            raise ValueError("catalog shock_id must be a one-dimensional integer array")
        if catmeta.nbytes > 256 * 1024**2:
            raise ValueError("catalog ID table exceeds this adapter's bounded in-memory index; supply a disk-backed ID adapter")
        catalog_ids = _slice(paths["catalog"], catmeta, 0, catmeta.shape[0]).astype(np.int64)
        catalog_ids.sort()
        if np.any(catalog_ids < 0) or np.any(catalog_ids[1:] == catalog_ids[:-1]):
            raise ValueError("catalog shock_id must contain unique nonnegative IDs")
    unit = result.get("position_unit")
    if unit not in {"km", "kpc", "Mpc"}:
        raise ValueError(f"snapshot {t}: unsupported position unit {unit!r}")
    factor = {"km": 1 / KPC_KM, "kpc": 1.0, "Mpc": 1000.0}[unit]
    stats = {"timestep": t, "retained_cells": n, "detected_shocks": 0, "valid_shocks": 0,
             "invalid_reasons": {}, "dissipation_feature": cfg.dissipation_feature,
             "catalog": "None (build_catalog=False)" if catalog_ids is None else "ID joined",
             "join": "center_index -> verified implicit dense retained-row ID, from user-supplied same-call producer",
             "position_unit_input": unit, "nonpositive_dissipation": 0,
             "normal_orientation": "signed upstream-to-downstream temperature gradient; normalized without sign flips"}
    pieces = {name: [] for name in ("ids", "pos", "normal", "mach", "cell_size", "dissipation")}
    for lo in range(0, n, cfg.chunk_rows):
        hi = min(n, lo + cfg.chunk_rows)
        shock = _slice(paths["result"], needed["shock"], lo, hi)
        rows = np.flatnonzero(shock) + lo
        if not rows.size:
            continue
        stats["detected_shocks"] += int(rows.size)
        ids = _rows(paths["result"], needed["center_index"], rows).astype(np.int64)
        if not np.array_equal(ids, rows):
            raise ValueError(f"snapshot {t}: accepted center_index is not its retained-row identifier; dense join refused")
        node_key(t, ids)  # Includes int64 encoding bounds.
        if catalog_ids is not None:
            index = np.searchsorted(catalog_ids, ids)
            exists = index < len(catalog_ids)
            if np.any(~exists) or not np.array_equal(catalog_ids[index], ids):
                raise ValueError(f"snapshot {t}: catalog lacks detected center IDs")
        values = {name: _rows(paths["result"], needed[name], rows).astype(np.float64)
                  for name in ("pos", "normal", "mach", "dx")}
        energy = _rows(paths["dissipation"], diss_meta[cfg.dissipation_feature], ids).astype(np.float64)
        flux = energy if cfg.dissipation_feature == "flux" else _rows(paths["dissipation"], diss_meta["flux"], ids)
        total = energy if cfg.dissipation_feature == "total" else _rows(paths["dissipation"], diss_meta["total"], ids)
        area = _rows(paths["dissipation"], diss_meta["area"], ids)
        # A corruption/producer consistency check, not the basis for the ID join.
        finite_energy = np.isfinite(flux) & np.isfinite(total) & np.isfinite(area)
        if np.any(finite_energy & ~np.isclose(total, flux * area, rtol=1e-9, atol=0)):
            raise ValueError(f"snapshot {t}: total != flux*area; supplied producer contract does not fit saved quantities")
        norms = np.linalg.norm(values["normal"], axis=1)
        flags = {
            "position_nonfinite": ~np.all(np.isfinite(values["pos"]), axis=1),
            "normal_invalid": ~np.isfinite(norms) | (norms <= 0),
            "mach_invalid": ~np.isfinite(values["mach"]) | (values["mach"] <= 0),
            "cell_size_invalid": ~np.isfinite(values["dx"]) | (values["dx"] <= 0),
        }
        if cfg.reject_mach_inconsistent and result.get("mach_consistent") is not None:
            info = array_metadata(result["mach_consistent"])
            if info.shape != (n,):
                raise ValueError("malformed mach_consistent field")
            flags["mach_inconsistent"] = ~_rows(paths["result"], info, rows).astype(bool)
        invalid = np.zeros(rows.size, dtype=bool)
        for reason, mask in flags.items():
            stats["invalid_reasons"][reason] = stats["invalid_reasons"].get(reason, 0) + int(mask.sum())
            invalid |= mask
        if invalid_sink is not None:
            for index in np.flatnonzero(invalid):
                invalid_sink.writerow({"timestep": t, "shock_id": int(ids[index]),
                                       "reasons": "|".join(name for name, mask in flags.items() if mask[index])})
        valid = ~invalid
        stats["valid_shocks"] += int(valid.sum())
        stats["nonpositive_dissipation"] += int(np.count_nonzero(valid & (~np.isfinite(energy) | (energy <= 0))))
        if meta is None:
            continue  # Preflight validates values/IDs; does not invent aexp/time.
        pos, dx = values["pos"][valid] * factor, values["dx"][valid] * factor
        if cfg.coordinate_frame == "physical":
            pos, dx = pos / meta.aexp, dx / meta.aexp
        if cfg.periodic:
            pos %= meta.box_comoving_kpc
        pieces["ids"].append(ids[valid])
        pieces["pos"].append(pos)
        pieces["normal"].append(values["normal"][valid] / norms[valid, None])
        pieces["mach"].append(values["mach"][valid])
        pieces["cell_size"].append(dx)
        pieces["dissipation"].append(energy[valid])
    if catalog_ids is not None and len(catalog_ids) != stats["detected_shocks"]:
        raise ValueError("identified catalog contains IDs that are not accepted result centers")
    if meta is None:
        return None, stats
    compact = {}
    for name, chunks in pieces.items():
        shape = (0, 3) if name in {"pos", "normal"} else (0,)
        compact[name] = np.concatenate(chunks) if chunks else np.empty(shape, dtype=np.int64 if name == "ids" else np.float64)
        chunks.clear()
    return Snapshot(t, meta.aexp, meta.time_gyr, **compact), stats
