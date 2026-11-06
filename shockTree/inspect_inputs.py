"""Discover and inspect saved inputs before adopting any temporal ID convention.

Run from the repository root, for example:
    python -m shockTree.inspect_inputs --input-dir /path/to/NC_shock

Only selected snapshots are opened. Binary pickle arrays are mapped read-only;
accepted-center IDs are audited in chunks. At most two compact ID lists are
retained. No matching, position prediction, row-order join, or tree is produced.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import config
from .pickle_metadata import array_metadata, object_fields, read_metadata


KINDS = ("result", "dissipation", "catalog")


def discover(input_dir: Path, start: int, end: int):
    if start < 0 or end < start:
        raise ValueError("invalid snapshot interval")
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    files = {}
    excluded = set()
    hints = (config.RESULT_PATTERN, config.DISSIPATION_PATTERN, config.CATALOG_PATTERN)
    paths = set()
    for pattern in hints:
        paths.update(input_dir.glob(pattern))
    # Resolve flattened downloads and prefixes/suffix widths differing from hints.
    paths.update(input_dir.rglob("*.pkl"))
    paths.update(input_dir.rglob("*.pickle"))
    for path in sorted(paths):
        if not path.is_file():
            continue
        kind_matches = [k for k in KINDS if re.search(rf"(?:^|[_-]){k}(?:[_-]|$)", path.stem)]
        if len(kind_matches) != 1:
            continue
        numbers = re.findall(r"\d+", path.stem)
        if len(numbers) != 1:
            raise ValueError(f"ambiguous snapshot in filename; explicit manifest needed: {path}")
        timestep = int(numbers[0])
        if not start <= timestep <= end:
            excluded.add(timestep)
            continue
        kind = kind_matches[0]
        row = files.setdefault(timestep, {})
        resolved = path.resolve()
        if kind in row and row[kind] != resolved:
            raise ValueError(f"multiple {kind} files for snapshot {timestep}; explicit manifest needed")
        row[kind] = resolved
    if not files:
        raise ValueError("no selected ShockFinder files discovered")
    return dict(sorted(files.items())), sorted(excluded)


def schema(fields):
    output = {}
    for name, value in fields.items():
        try:
            metadata = array_metadata(value)
        except ValueError:
            output[name] = {"type": type(value).__name__, "value": value}
        else:
            output[name] = {
                "type": "numpy.ndarray", "shape": list(metadata.shape),
                "dtype": metadata.dtype.str, "order": "F" if metadata.fortran_order else "C",
                "payload_bytes": metadata.nbytes, "payload_offset": metadata.offset,
            }
    return output


@contextmanager
def mapped(path, fields, names):
    arrays = {}
    try:
        for name in names:
            arrays[name] = array_metadata(fields[name]).map(path)
        yield arrays
    finally:
        for value in arrays.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        arrays.clear()


def center_ids(path, fields, chunk_rows):
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    n = array_metadata(fields["shock"]).shape
    if len(n) != 1:
        raise ValueError("shock mask is not one dimensional")
    n = n[0]
    if array_metadata(fields["shock"]).dtype != np.dtype(bool):
        raise ValueError("shock mask is not boolean")
    center_meta = array_metadata(fields["center_index"])
    if center_meta.shape != (n,) or center_meta.dtype.kind not in "iu":
        raise ValueError("center_index is not an integer retained-row array")
    chunks = []
    matches_rows = True
    has_negative = False
    with mapped(path, fields, ("shock", "center_index")) as arrays:
        for lo in range(0, n, chunk_rows):
            hi = min(lo + chunk_rows, n)
            rows = np.flatnonzero(arrays["shock"][lo:hi]) + lo
            ids = np.array(arrays["center_index"][rows], dtype=np.int64)
            matches_rows &= np.array_equal(ids, rows)
            has_negative |= bool(np.any(ids < 0))
            if ids.size:
                chunks.append(ids)
    ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    del chunks
    unique = matches_rows or np.unique(ids).size == ids.size
    return ids, {"retained_cells": n, "detected_shocks": int(ids.size),
                 "center_ids_equal_retained_row": bool(matches_rows),
                 "center_ids_unique_within_snapshot": bool(unique),
                 "negative_center_ids": has_negative,
                 "center_id_min": int(ids.min()) if ids.size else None,
                 "center_id_max": int(ids.max()) if ids.size else None}


def sample_measurements(path, fields, ids, maximum=1024):
    if not ids.size:
        return {"sample_count": 0}
    if not np.all(ids[1:] > ids[:-1]):
        return {"sample_count": 0, "reason": "center IDs are not verified retained rows"}
    rows = ids[np.linspace(0, ids.size - 1, min(maximum, ids.size), dtype=np.int64)]
    names = [name for name in ("mach", "pos", "normal", "dx", "level", "upstream_index", "downstream_index")
             if name in fields]
    with mapped(path, fields, names) as arrays:
        values = {name: np.asarray(array[rows]).copy() for name, array in arrays.items()}
        lengths = np.linalg.norm(values["normal"], axis=1)
        valid = np.all(np.isfinite(values["pos"]), axis=1) & np.isfinite(lengths) & (lengths > 0)
        valid &= np.isfinite(values["mach"]) & (values["mach"] > 0)
        valid &= np.isfinite(values["dx"]) & (values["dx"] > 0)
        result = {"sample_count": int(rows.size), "sampling": "evenly spaced accepted-center IDs",
                  "sample_valid_geometry_mach_dx": int(valid.sum()),
                  "sample_normal_norm_min": float(np.min(lengths)),
                  "sample_normal_norm_max": float(np.max(lengths)),
                  "sample_normals_unit_within_tolerance": int(np.count_nonzero(np.abs(lengths - 1) <= config.NORMAL_TOLERANCE)),
                  "sample_mach_min": float(np.min(values["mach"])),
                  "sample_mach_max": float(np.max(values["mach"])),
                  "sample_dx_min": float(np.min(values["dx"])),
                  "sample_dx_max": float(np.max(values["dx"])),
                  "sample_position_min": np.min(values["pos"], axis=0).tolist(),
                  "sample_position_max": np.max(values["pos"], axis=0).tolist()}
        if "level" in values:
            result.update(sample_level_min=int(values["level"].min()), sample_level_max=int(values["level"].max()))
        if "upstream_index" in values and "downstream_index" in values:
            up, down = values["upstream_index"], values["downstream_index"]
            n = array_metadata(fields["mach"]).shape[0]
            usable = (up >= 0) & (up < n) & (down >= 0) & (down < n) & np.isfinite(lengths) & (lengths > 0)
            # This is a source-convention check only. No periodicity is assumed.
            displacement = arrays["pos"][down[usable]] - arrays["pos"][up[usable]]
            dot = np.sum(displacement * values["normal"][usable], axis=1)
            result["sample_endpoint_orientation"] = {
                "tested": int(usable.sum()), "positive_upstream_to_downstream_projection": int(np.count_nonzero(dot > 0)),
                "negative_projection": int(np.count_nonzero(dot < 0)),
                "zero_projection": int(np.count_nonzero(dot == 0)), "periodic_unwrapping_applied": False,
            }
    return result


def inspect(input_dir, output_dir, start, end, chunk_rows):
    discovered, excluded = discover(input_dir, start, end)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"input_dir": str(input_dir.resolve()), "snapshot_start": start, "snapshot_end": end,
              "selected_snapshots": list(discovered), "outside_range_not_loaded": excluded,
              "snapshot_number_convention": "integer parsed from the sole digit group in each filename; selected snapshots sorted numerically",
              "snapshots": [], "id_collisions": [], "blockers": [],
              "scope": "shock-center AMR cell detections, not persistent material cells or aggregated shock objects",
              "adopted_shock_id_field": config.SHOCK_ID_FIELD,
              "id_reference_convention": config.ID_REFERENCE_CONVENTION,
              "row_order_join_performed": False, "tree_created": False}
    previous_ids = None
    previous_timestep = None
    for timestep, paths in discovered.items():
        entry = {"timestep": timestep, "files": {k: str(v) for k, v in paths.items()}, "objects": {}}
        missing = set(KINDS) - paths.keys()
        if missing:
            report["blockers"].append(f"snapshot {timestep}: missing {sorted(missing)} files")
        result_fields = None
        current_ids = None
        for kind in KINDS:
            if kind not in paths:
                continue
            path = paths[kind]
            object_type, fields = object_fields(read_metadata(path))
            entry["objects"][kind] = {"type": object_type, "file_bytes": path.stat().st_size,
                                      "fields": schema(fields)}
            # Embedded metadata must agree, when it exists; absence is recorded.
            for name in ("timestep", "snapshot", "iout"):
                if name in fields and fields[name] != timestep:
                    raise ValueError(f"{path}: embedded {name} disagrees with filename")
            entry["objects"][kind]["embedded_snapshot_present"] = any(k in fields for k in ("timestep", "snapshot", "iout"))
            if kind == "result":
                result_fields = fields
                current_ids, counts = center_ids(path, fields, chunk_rows)
                entry.update(counts)
                if not counts["center_ids_unique_within_snapshot"] or counts["negative_center_ids"]:
                    report["blockers"].append(f"snapshot {timestep}: center IDs fail within-snapshot uniqueness/validity")
                if counts["center_ids_equal_retained_row"]:
                    entry["result_measurement_sample"] = sample_measurements(path, fields, current_ids)
                if config.SHOCK_ID_FIELD not in fields:
                    report["blockers"].append(f"snapshot {timestep}: configured ID field is absent")
                entry["position_unit"] = fields.get("position_unit")
                entry["aexp_field"] = next((k for k in ("aexp", "scale_factor") if k in fields), None)
                entry["physical_time_field"] = next((k for k in ("time_gyr", "t_BB", "time") if k in fields), None)
                if entry["aexp_field"] is None or entry["physical_time_field"] is None:
                    report["blockers"].append(f"snapshot {timestep}: scale factor and/or physical time metadata absent")
            if kind == "catalog" and object_type == "builtins.NoneType":
                entry["catalog_interpretation"] = "None is expected from the user-verified build_catalog=False producer; no catalog records to join"
            if kind == "dissipation" and not any(k in fields for k in ("shock_id", "center_index", "selected_indices", "input_cell_id")):
                entry["dissipation_identifier"] = "implicit dense retained-row ID, verified by the user-supplied analyze(...,compact=False) producer contract"
        if result_fields is not None and "dissipation" in entry["objects"]:
            shapes = {name: info.get("shape") for name, info in entry["objects"]["dissipation"]["fields"].items()}
            entry["dissipation_has_same_dense_row_count"] = all(shape == [entry["retained_cells"]] for shape in shapes.values())
            entry["dissipation_order_verified_by_producer_contract"] = bool(entry["dissipation_has_same_dense_row_count"] and entry.get("center_ids_equal_retained_row"))
            entry["dissipation_order_independently_serialized"] = False
            if not entry["dissipation_order_verified_by_producer_contract"]:
                report["blockers"].append(f"snapshot {timestep}: verified dense producer ID contract is not satisfied")
        if current_ids is not None:
            # Exact adjacent collision evidence; any one collision disproves global uniqueness.
            if previous_ids is not None:
                repeated = np.intersect1d(previous_ids, current_ids)
                report["id_collisions"].append({"snapshots": [previous_timestep, timestep],
                                                "repeated_center_ids": int(repeated.size),
                                                "examples": repeated[:10].tolist()})
                if repeated.size and config.ID_REFERENCE_CONVENTION is None:
                    report["blockers"].append("center_index values repeat across snapshots; a global reference convention requires user input")
                del repeated
            previous_ids, previous_timestep = current_ids, timestep
        report["snapshots"].append(entry)
        print(f"snapshot {timestep}: {entry.get('detected_shocks', 'unknown'):,} accepted centers" if "detected_shocks" in entry else f"snapshot {timestep}: no result", flush=True)
        gc.collect()
    report["blockers"].extend([
        "coordinate frame/origin and simulation periodic-box convention are not embedded; external metadata required",
        "only the locally discovered snapshots can be inspected; full selected-range coverage must be checked on the input host",
    ])
    report["blockers"] = list(dict.fromkeys(report["blockers"]))
    report["repository_conventions_not_file_metadata"] = {
        "normal": "unit temperature-gradient vector, upstream toward downstream; signed cosine must be retained",
        "position": "ShockFinder uses cell positions in result.position_unit; physical/comoving frame and extraction origin not serialized",
        "boundary": "repository uses open boundaries for extracted regions; this does not establish the NewCluster periodic box",
        "dissipation_units": {"flux": "erg/s/kpc2", "total": "erg/s", "area": "kpc2", "efficiency": "dimensionless", "sound_speed": "km/s"},
        "dissipation_order": "compute_dissipation allocates the same retained-row space as result; the saved files lack a row-ID table or source fingerprint",
        "shock_id_alias": "user authorized preserving existing result.center_index values in shock_id and (timestep<<32)|center_index in reference fields",
        "prediction": "upstream velocity is not stored; M*sound_speed cannot establish simulation-frame propagation",
    }
    with (output_dir / "inspection.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output_dir / "input_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestep", "result_path", "dissipation_path", "catalog_path"])
        writer.writeheader()
        for timestep, paths in discovered.items():
            writer.writerow({"timestep": timestep, **{f"{kind}_path": str(paths.get(kind, "")) for kind in KINDS}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(config.INPUT_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--snapshot-start", type=int, default=config.SNAPSHOT_START)
    parser.add_argument("--snapshot-end", type=int, default=config.SNAPSHOT_END)
    parser.add_argument("--chunk-rows", type=int, default=config.INSPECTION_CHUNK_ROWS)
    args = parser.parse_args()
    try:
        report = inspect(args.input_dir, args.output_dir, args.snapshot_start, args.snapshot_end, args.chunk_rows)
    except (ValueError, OSError, KeyError, pickle.UnpicklingError) as exc:
        parser.exit(2, f"Inspection failed: {exc}\n")
    print(f"Inspection written to {args.output_dir}; {len(report['blockers'])} blocking findings.")
    # A successful inspection with unresolved prerequisites must not signal readiness.
    if report["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
