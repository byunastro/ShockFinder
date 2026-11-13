"""Export exact RUR header metadata on the NewCluster simulation host.

Only snapshot headers are needed. No cells, ShockFinder detection, or physical
algorithm are loaded or run. The supplied repository exporter establishes
RUR's params['z'], params['age'] and unit['kpc'] conventions for NewCluster.
Coordinate frame/origin and periodicity remain explicit producer declarations.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import INPUT_DIR, SNAPSHOT_END, SNAPSHOT_START
from .diagnostics import write_json
from .inspect_inputs import discover


def snapshot_row(timestep, params, unit_kpc, periodic, box_override=None):
    redshift = float(params["z"])
    age = float(params["age"])
    aexp = float(params.get("aexp", 1 / (1 + redshift)))
    if not np.isfinite(redshift) or redshift <= -1 or not np.isfinite(age) or age < 0 or not np.isfinite(aexp) or aexp <= 0:
        raise ValueError(f"snapshot {timestep}: invalid RUR redshift/age/aexp")
    if not np.isclose(aexp, 1 / (1 + redshift), rtol=1e-8, atol=0):
        raise ValueError(f"snapshot {timestep}: aexp and redshift disagree")
    row = {"timestep": timestep, "aexp": aexp, "time_gyr": age,
           "box_x_comoving_kpc": "", "box_y_comoving_kpc": "", "box_z_comoving_kpc": ""}
    if periodic:
        if box_override is not None:
            box = np.asarray(box_override, dtype=float)
        else:
            unit_kpc = float(unit_kpc)
            if not np.isfinite(unit_kpc) or unit_kpc <= 0 or "boxlen" not in params:
                raise ValueError("periodic box requires RUR boxlen/unit['kpc'] or --box-comoving-kpc")
            # RUR code lengths / unit['kpc'] are physical kpc in the NC workflow.
            box = np.full(3, float(params["boxlen"]) / unit_kpc / aexp)
        if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
            raise ValueError("invalid comoving periodic box")
        for axis, value in zip("xyz", box):
            row[f"box_{axis}_comoving_kpc"] = float(value)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-repo", type=Path, default=Path("/storage7/NewCluster2"))
    parser.add_argument("--input-dir", type=Path, default=Path(INPUT_DIR))
    parser.add_argument("--output", type=Path, default=Path("shockTree/snapshot_metadata.csv"))
    parser.add_argument("--snapshot-start", type=int, default=SNAPSHOT_START)
    parser.add_argument("--snapshot-end", type=int, default=SNAPSHOT_END)
    parser.add_argument("--coordinate-frame", required=True, choices=["physical", "comoving"])
    parser.add_argument("--coordinate-origin", required=True, choices=["same_simulation_origin", "common_unwrapped_origin"])
    parser.add_argument("--periodic", required=True, choices=["yes", "no"])
    parser.add_argument("--box-comoving-kpc", type=float, nargs=3)
    args = parser.parse_args()
    periodic = args.periodic == "yes"
    if periodic and args.coordinate_origin == "common_unwrapped_origin":
        parser.error("a common unwrapped region must use --periodic no")
    from rur import uri

    files, _ = discover(args.input_dir, args.snapshot_start, args.snapshot_end)
    rows = []
    for t in files:
        snap = uri.RamsesSnapshot(repo=str(args.sim_repo), iout=t, mode="nc", longint=False)
        rows.append(snapshot_row(t, snap.params, snap.unit["kpc"], periodic, args.box_comoving_kpc))
        del snap
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestep", "aexp", "time_gyr", "box_x_comoving_kpc", "box_y_comoving_kpc", "box_z_comoving_kpc"])
        writer.writeheader()
        writer.writerows(rows)
    declarations = {"sim_repo": str(args.sim_repo), "input_dir": str(args.input_dir), "selected_snapshots": list(files),
                    "coordinate_frame": args.coordinate_frame, "coordinate_origin": args.coordinate_origin, "periodic": periodic,
                    "time_source": "RUR snapshot params['age'] (Gyr)", "aexp_source": "params['aexp'], checked against 1/(1+params['z'])",
                    "box_source": "explicit override" if args.box_comoving_kpc else "boxlen / unit['kpc'] / aexp",
                    "declarations": "frame/origin/periodicity supplied explicitly by the operator; header export cannot infer saved-region recentering"}
    write_json(args.output.with_suffix(".provenance.json"), declarations)
    print(f"Wrote {len(rows)} header rows to {args.output}. Copy the frame/origin/periodic declarations into pipeline configuration.")


if __name__ == "__main__":
    main()
