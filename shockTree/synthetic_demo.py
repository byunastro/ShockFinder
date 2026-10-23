"""Generate and process a labelled synthetic expanding/splitting/merging example.

This creates synthetic saved dictionaries, not simulation outputs. It does
not invoke ShockFinder. The resulting NPZ and plots demonstrate the pipeline
and must not be interpreted as real NewCluster associations.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from .config import MatchOptions, PipelineConfig
from .pipeline import build


def make_demo(directory):
    directory = Path(directory)
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    theta = np.arange(8) * np.pi / 4
    for ordinal, t in enumerate((709, 710, 712, 714)):
        directions = np.column_stack([np.cos(theta + .01 * ordinal), np.sin(theta + .01 * ordinal), np.zeros(8)])
        positions = list((5 + .3 * ordinal) * directions)
        normals = list(directions)
        positions.append([15., 0, 0])
        normals.append([0., 1, 0])
        if ordinal == 1:  # A competing split candidate, followed by a merge.
            positions.append([15.4, 0, 0])
            normals.append([0., 1, 0])
        if ordinal == 3:  # An appearing detection with no candidate progenitor.
            positions.append([30., 0, 0])
            normals.append([0., 1, 0])
        positions, normals = np.asarray(positions), np.asarray(normals)
        # Changed retained row IDs model AMR sampling, not persistent cells.
        ids = np.arange(len(positions), dtype=np.int64) * 3 + ordinal
        n = int(ids.max()) + 1
        shock = np.zeros(n, dtype=bool)
        shock[ids] = True
        center = np.full(n, -1, dtype=np.int32)
        center[ids] = ids
        pos, normal, mach = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
        pos[ids], normal[ids], mach[ids] = positions, normals * 2, 2 + .05 * ordinal
        result = {"shock": shock, "center_index": center, "selected_indices": np.arange(n, dtype=np.int32),
                  "mach": mach, "pos": np.asfortranarray(pos), "normal": np.asfortranarray(normal),
                  "dx": np.full(n, .05 / (2 if ordinal == 1 else 1)), "position_unit": "kpc"}
        flux, area = np.zeros(n), np.full(n, .0025)
        flux[ids] = 3 + .1 * ordinal
        dissipation = {"flux": flux, "total": flux * area, "area": area,
                       "efficiency": np.full(n, .1), "sound_speed": np.full(n, 100.)}
        for kind, value in (("result", result), ("dissipation", dissipation), ("catalog", None)):
            (inputs / f"{kind}_{t:05d}.pkl").write_bytes(pickle.dumps(value, protocol=4))
    metadata = directory / "synthetic_metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestep", "aexp", "time_gyr"])
        writer.writeheader()
        writer.writerows({"timestep": t, "aexp": 1.0, "time_gyr": ordinal * .01}
                         for ordinal, t in enumerate((709, 710, 712, 714)))
    cfg = PipelineConfig(input_dir=str(inputs), output_path=str(directory / "synthetic_tree.npz"),
                         metadata_path=str(metadata), snapshot_start=709, snapshot_end=714,
                         coordinate_frame="physical", coordinate_origin="same_simulation_origin", periodic=False,
                         diagnostic_label="Synthetic demonstration", matching=MatchOptions(max_speed_kms=200.))
    return cfg, build(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "demo_output")
    args = parser.parse_args()
    cfg, report = make_demo(args.output_dir)
    with np.load(cfg.output_path, allow_pickle=False) as archive:
        tree = archive["shock_tree"]
    if len(tree):
        shock = tree[0]
        branch = np.sort(tree[tree["last"] == shock["last"]], order="timestep")
        print("Synthetic branch timesteps:", branch["timestep"].tolist())
    print("Synthetic validation:", report["validation"])
    print("Synthetic output:", cfg.output_path)


if __name__ == "__main__":
    main()
