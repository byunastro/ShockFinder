"""Build a compact primary shock tree from already saved dense detections."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from .config import PipelineConfig, load_config, shock_tree_dtype
from .diagnostics import branch_diagnostics, pair_diagnostics, write_json
from .inputs import _slice, load_snapshot, read_metadata_table, selected_files
from .matching import match_snapshots
from .pickle_metadata import array_metadata, object_fields, read_metadata
from .tree import BRANCH_STATE_DTYPE, append_snapshot, construct_branches, link_pair, validate_tree


def _count_centers(path, chunk_rows):
    _, fields = object_fields(read_metadata(path))
    mask = array_metadata(fields["shock"])
    if len(mask.shape) != 1 or mask.dtype != np.dtype(bool):
        raise ValueError("result.shock must be a one-dimensional boolean array")
    return sum(int(np.count_nonzero(_slice(path, mask, lo, min(mask.shape[0], lo + chunk_rows))))
               for lo in range(0, mask.shape[0], chunk_rows))


def save_compact(path, tree):
    """Stream one array into a compressed archive and atomically install it."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, shock_tree=tree)
        # Validate the archive/header without loading the whole range into RAM.
        with zipfile.ZipFile(temporary) as archive:
            if archive.namelist() != ["shock_tree.npy"]:
                raise ValueError("main archive contains unexpected arrays")
            with archive.open("shock_tree.npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported saved NPY header version {version}")
                if shape != tree.shape or dtype != shock_tree_dtype or fortran:
                    raise ValueError("saved archive header differs from the validated tree")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preflight(cfg):
    cfg.validate(require_physics=False)
    files, excluded = selected_files(cfg)
    output = Path(cfg.output_path).with_suffix("")
    directory = output.parent / (output.name + "_preflight")
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "input_validation_only", "tree_created": False, "snapshots": [],
              "selected_snapshots": list(files), "outside_range_not_loaded": excluded,
              "missing_physics": "aexp/time metadata and explicit frame/origin/boundary declaration are required for matching"}
    with (directory / "invalid_records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestep", "shock_id", "reasons"])
        writer.writeheader()
        for t, paths in files.items():
            _, stats = load_snapshot(paths, None, cfg, writer, timestep=t)
            report["snapshots"].append(stats)
            print(f"snapshot {t}: {stats['valid_shocks']:,}/{stats['detected_shocks']:,} valid detections", flush=True)
    write_json(directory / "input_validation.json", report)
    return report, directory


def build(cfg: PipelineConfig):
    cfg.validate()
    files, excluded = selected_files(cfg)
    metadata_path = Path(cfg.metadata_path)
    if not metadata_path.is_absolute():
        metadata_path = Path(cfg.input_dir) / metadata_path
    metadata = read_metadata_table(metadata_path, list(files), cfg)
    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output.with_suffix("")
    diagnostic_dir = diagnostic_dir.parent / (diagnostic_dir.name + "_diagnostics")
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "selected_snapshots": list(files), "outside_range_not_loaded": excluded,
              "requested_bounds": [cfg.snapshot_start, cfg.snapshot_end], "snapshots": [], "pairs": [],
              "reference_convention": "(int64(timestep) << 32) | int64(shock_id); original center_index remains shock_id",
              "isolated_node_convention": "fat=son=-1; first=last=node_key(timestep, shock_id)",
              "reciprocity": "parent.son == node_key(child); child.fat == node_key(parent)",
              "output_position_unit": "physical kpc", "matching_frame": "comoving kpc with physical Gyr times",
              "scope": "individual shock-center AMR detections in physical space, not persistent material cells",
              "range_boundaries": "first/last refer only to selected snapshots",
              "configuration": cfg.to_dict(), "scores_calibrated": False}
    started = time.monotonic()
    try:
        # A counting pass reads only masks, one snapshot at a time. Capacity can
        # exceed final length because invalid records are excluded during loading.
        counts = {t: _count_centers(paths["result"], cfg.chunk_rows) for t, paths in files.items()}
        capacity = sum(counts.values())
        report["staging_capacity_nodes"] = capacity
        report["staging_tree_bytes"] = capacity * shock_tree_dtype.itemsize
        # Both disk-backed state and an incompressible final archive may coexist.
        required_space = int(capacity * (2.01 * shock_tree_dtype.itemsize + BRANCH_STATE_DTYPE.itemsize)) + 1024**2
        report["conservative_required_free_bytes"] = required_space
        if shutil.disk_usage(output.parent).free < required_space:
            raise ValueError(f"insufficient scratch/output space; conservative requirement is {required_space:,} bytes")
        if len(files) >= 2 and cfg.max_plot_pairs:
            plot_ordinals = set(np.linspace(0, len(files) - 2, min(cfg.max_plot_pairs, len(files) - 1), dtype=int))
        else:
            plot_ordinals = set()
        with tempfile.TemporaryDirectory(prefix=".shock_tree_work_", dir=output.parent) as work:
            work = Path(work)
            # open_memmap accepts empty arrays but there is no data mapping to close.
            tree = np.lib.format.open_memmap(work / "tree.npy", mode="w+", dtype=shock_tree_dtype, shape=(capacity,))
            state = np.lib.format.open_memmap(work / "branch_state.npy", mode="w+", dtype=BRANCH_STATE_DTYPE, shape=(capacity,))
            segments = {}
            cursor = 0
            previous = previous_range = None
            with (diagnostic_dir / "invalid_records.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["timestep", "shock_id", "reasons"])
                writer.writeheader()
                for ordinal, (t, paths) in enumerate(files.items()):
                    snapshot, stats = load_snapshot(paths, metadata[t], cfg, writer)
                    report["snapshots"].append(stats)
                    current_range = append_snapshot(tree, cursor, snapshot)
                    cursor = current_range[1]
                    segments[t] = current_range
                    print(f"snapshot {t}: {len(snapshot):,}/{stats['detected_shocks']:,} valid detections", flush=True)
                    if previous is not None:
                        pair_started = time.monotonic()
                        result = match_snapshots(previous, snapshot, cfg.matching, metadata[t].box_comoving_kpc)
                        link_pair(tree, previous_range, current_range, previous, snapshot, result)
                        result.stats["elapsed_wall_seconds"] = time.monotonic() - pair_started
                        if cfg.diagnostics:
                            pair_diagnostics(diagnostic_dir / f"pair_{previous.timestep:05d}_{t:05d}", previous, snapshot,
                                             result, cfg, metadata[t].box_comoving_kpc, ordinal - 1 in plot_ordinals)
                        report["pairs"].append(result.stats)
                        print(f"pair {previous.timestep} → {t}: {len(result.accepted):,} primary links, {len(result.edges):,} eligible edges", flush=True)
                        del result
                    previous, previous_range = snapshot, current_range
                    # No dense source arrays or previous-previous snapshot remain.
                    write_json(diagnostic_dir / "run.json", report)
            del previous, snapshot
            valid_tree, valid_state = tree[:cursor], state[:cursor]
            construct_branches(valid_tree, segments, valid_state, cfg.chunk_rows)
            report["validation"] = validate_tree(valid_tree, segments, metadata, cfg.chunk_rows)
            if cfg.diagnostics:
                report["branch_statistics"] = branch_diagnostics(diagnostic_dir, valid_tree, valid_state, segments, metadata, cfg)
                with (diagnostic_dir / "snapshot_statistics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["timestep", "aexp", "time_gyr", "detected_shocks", "valid_shocks"])
                    writer.writeheader()
                    writer.writerows({"timestep": row["timestep"], "aexp": metadata[row["timestep"]].aexp,
                                      "time_gyr": metadata[row["timestep"]].time_gyr,
                                      "detected_shocks": row["detected_shocks"], "valid_shocks": row["valid_shocks"]}
                                     for row in report["snapshots"])
            tree.flush()
            save_compact(output, valid_tree)
            del valid_tree, valid_state
            tree._mmap.close()
            state._mmap.close()
        report["status"] = "complete"
        report["elapsed_wall_seconds"] = time.monotonic() - started
        report["output_path"] = str(output.resolve())
        report["compressed_bytes"] = output.stat().st_size
        write_json(diagnostic_dir / "run.json", report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["elapsed_wall_seconds"] = time.monotonic() - started
        write_json(diagnostic_dir / "run.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-dir")
    parser.add_argument("--output-path")
    parser.add_argument("--metadata-path")
    parser.add_argument("--manifest-path")
    parser.add_argument("--snapshot-start", type=int)
    parser.add_argument("--snapshot-end", type=int)
    parser.add_argument("--preflight", action="store_true", help="validate actual IDs/quantities without time/frame metadata; no matching")
    parser.add_argument("--short-interval", action="store_true", help="build only the explicitly configured test interval")
    parser.add_argument("--trial-start", type=int, help="short core-passage interval to test before the full range")
    parser.add_argument("--trial-end", type=int)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        for name in ("input_dir", "output_path", "metadata_path", "manifest_path", "snapshot_start", "snapshot_end"):
            value = getattr(args, name)
            if value is not None:
                setattr(cfg, name, value)
        if args.preflight:
            _, directory = preflight(cfg)
            print(f"Input validation saved to {directory}; temporal matching was not run.")
            return
        cfg.validate()
        if not args.short_interval:
            if args.trial_start is None or args.trial_end is None:
                raise ValueError("specify --trial-start/--trial-end around measured core passage; or use --short-interval for the initial test")
            if not cfg.snapshot_start <= args.trial_start < args.trial_end <= cfg.snapshot_end:
                raise ValueError("trial bounds must lie inside the full requested range")
            trial = copy.deepcopy(cfg)
            trial.snapshot_start, trial.snapshot_end = args.trial_start, args.trial_end
            path = Path(cfg.output_path)
            trial.output_path = str(path.with_name(path.stem + "_trial.npz"))
            if len(selected_files(trial)[0]) < 2:
                raise ValueError("trial interval contains fewer than two selected snapshots")
            print("Testing the short core-passage interval first.", flush=True)
            build(trial)
            print("Trial invariants passed. Processing the full selected range.", flush=True)
        result = build(cfg)
        print(json.dumps({"output": result["output_path"], "validation": result["validation"]}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"shockTree stopped: {exc}\n")


if __name__ == "__main__":
    main()
