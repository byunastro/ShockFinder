"""End-to-end example using SAVED ShockFinder outputs and galaxy histories.

Run from the repository root, or put it on Python's import path::

    from rur import utool
    from examples.merger_shock_stripping import find_merger_shock_stripped_galaxies

    # Already prepared in memory; no galaxy pickle format is assumed here.
    # galaxy_histories maps a persistent branch ID to snapshot-aligned arrays.
    # analysis_snapshots should cover the interval before/after possible losses.
    products = find_merger_shock_stripped_galaxies(
        galaxy_histories, cluster_info, analysis_snapshots,
        shock_output_dir="/storage1/byunkh/NC_shock/shock_output",
        cache_root="local_work/merger_shock_stripping/input_cache",
        output_dir="local_work/merger_shock_stripping/results",
        load_saved=utool.load,
        coverage_by_snapshot=verified_shock_coverage,
        stripping_options={"tracers": ["m_gas_r90", "HI_gas_mass_r90"],
                           "min_fractional_loss": .30,
                           "min_loss_rate_gyr": 2., "window_gyr": .15},
    )
    analysis = products["analysis"]
    candidates = products["candidates"]
    candidate_ids = products["candidate_galaxy_ids"]

Required history declarations::

    history["units"] = {"time": "Gyr", "position": "kpc", "radius": "kpc",
                        "velocity": "km/s", "coordinate_frame": "physical"}

Declare the ACTUAL units; physical km/Mpc positions and radii are also supported.
Provide iout, t_BB, gal_cen, gas_radius (or flagged r90 proxy), and every gas
column explicitly named in stripping_options["tracers"] (default: m_gas_r90).
Every selected tracer is analyzed independently; no tracer count is required.
P_ram is optional and used only for diagnostics. vx/vy/vz optionally provide
motion angles. Preserve additional fields. Supply per-tracer valid/resolved
masks and mass_limits when
known. Missing or unresolved measurements remain flagged. hmid can change;
the outer mapping key must identify the same galaxy throughout its history.

cluster_info contains the full two-center history and simulation times, even
if only some outputs have shock data. Its physical position_unit defaults to
kpc. Fill secondary centers/radii with NaN after its tree branch terminates.

verified_shock_coverage is optional. An entry may be
{iout: {"complete": True, "covered_galaxy_ids": [branch_id, ...]}} ONLY when the
saved shock search covers those galaxies' positions AND intervening trajectories.
Exclude trajectories near the extraction faces and boundary-excluded fronts
from that coverage assertion. Omitted coverage cannot establish non-exposure.

The first cache conversion still needs the original pickle objects in RAM.
Subsequent runs open memory-mapped detected-cell inputs. No detector or shock
physics is run. All four galaxy categories are retained. Selected galaxies
are evidence-based candidates, not causally confirmed stripped galaxies.

cluster_info may be loaded with the supplied reader from
"/Users/byeongyeonghwan/NC_shock/cluster_info.pkl". The extraction is an OPEN
3000-kpc cube centered on the two-center midpoint before output 870 and on the
primary thereafter. Boundary-intersecting fronts remain recorded but excluded.
Set merger_options={"neighbor_workers": 4} to reuse a Fortran worker pool
throughout this function. For multiprocessing, call it from a Python script
under ``if __name__ == "__main__":``. The default is one worker.
"""

from __future__ import annotations

import gc
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from examples.galaxy import galaxy_stripping_catalog
from examples.shock_catalog import (
    cache_merger_shock_inputs,
    load_merger_shock_inputs,
    merger_neighbor_pool,
    merger_shock_catalog,
)


def find_merger_shock_stripped_galaxies(
    galaxy_histories,
    cluster_info,
    snapshots,
    *,
    shock_output_dir,
    cache_root,
    output_dir,
    load_saved=None,
    coverage_by_snapshot=None,
    merger_options=None,
    stripping_options=None,
    make_plots=False,
):
    """Build snapshot merger fronts, measure proximity/loss, and select candidates.

    Snapshot fronts have no cross-output identity. Their proximity does not
    establish a tracked crossing; provide independently tracked members to
    galaxy_stripping_catalog when temporal crossing evidence is required.

    load_saved is the existing input reader, e.g. rur.utool.load. It receives a
    filename string and is needed only when an output has no completed cache.
    Input caches are reused and never overwritten; delete/rebuild them yourself
    if the underlying saved inputs change. Include endpoints to support signed
    upstream/downstream geometry. Default merger evidence and stripping cuts
    come from _MergerOptions and StrippingOptions; the two option dictionaries
    override them. Sensitivity sweeps remain enabled.

    make_plots=True saves diagnostics for ALL galaxies and the population.
    For a large sample, leave it False and use plot_galaxy_stripping(analysis,
    galaxy_id) or plot_stripping_population(analysis) on the desired subset.

    Returns analysis, independent merger_catalogs keyed by snapshot, candidate
    rows/IDs, and the saved merger-catalog path. The other categories and all
    uncertain gas events/encounters remain in analysis. Existing center IDs
    and full membership can be recovered with get_merger_shock_members().
    """
    snapshots = np.asarray(list(snapshots))
    if (snapshots.ndim != 1 or snapshots.dtype.kind not in "iu" or not snapshots.size
            or len(np.unique(snapshots)) != len(snapshots)):
        raise ValueError("snapshots must be nonempty unique integer output numbers")
    snapshots = sorted(int(s) for s in snapshots)
    shock_output_dir, cache_root, output_dir = map(
        Path, (shock_output_dir, cache_root, output_dir))
    coverage = {} if coverage_by_snapshot is None else coverage_by_snapshot

    # Prepare snapshot options without mutating the caller's cluster dictionary.
    centers = dict(cluster_info)
    options = dict(centers.get("merger_shock_options", {}))
    options.setdefault("expand_candidate_cells", False)
    if merger_options is not None:
        options.update(merger_options)

    # The supplied ShockFinder inputs are open 3000-kpc extracted cubes.
    # Their length must NEVER become a periodic distance/wrapping period.
    if options.get("box_size_kpc") is not None:
        raise ValueError("the extracted ShockFinder cube is open; box_size_kpc must be None")
    centers["box_size"] = None
    centers["merger_shock_options"] = options

    catalogs = []
    workers = options.get("neighbor_workers", 1)
    pool_context = merger_neighbor_pool(workers) if workers != 1 else nullcontext()
    with pool_context:
        for iout in snapshots:
            directory = cache_root/f"snapshot_{iout:05d}"
            if (directory/"metadata.json").is_file():
                inputs = load_merger_shock_inputs(directory)
            else:
                if load_saved is None:
                    raise ValueError("supply load_saved for outputs without a completed input cache")
                result_path = shock_output_dir/f"result_{iout:05d}.pkl"
                dissipation_path = shock_output_dir/f"dissipation_{iout:05d}.pkl"
                result = load_saved(str(result_path))
                dissipation = load_saved(str(dissipation_path))
                try:
                    inputs = cache_merger_shock_inputs(
                        iout, result, dissipation, directory, include_endpoints=True,
                        provenance={"result": str(result_path.resolve()),
                                    "dissipation": str(dissipation_path.resolve())})
                finally:
                    del result, dissipation
                    gc.collect()
            catalog = merger_shock_catalog(iout, inputs, None, centers)
            catalogs.append(catalog)
            del inputs, catalog

    catalog_by_snapshot = {c["iout"]: c for c in catalogs}
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir/"merger_catalogs.pkl"
    with catalog_path.open("wb") as stream:
        pickle.dump(catalog_by_snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load_shock_product(iout):
        # Missing outputs are unknown exposure. Only this output's saved cell
        # arrays are opened; galaxy.py keeps compact previous/current fronts.
        catalog = catalog_by_snapshot.get(iout)
        if catalog is None:
            return None
        directory = cache_root/f"snapshot_{iout:05d}"
        checked = coverage.get(iout, {})
        return {"catalog": catalog, "result": load_merger_shock_inputs(directory),
                "dissipation": None, "complete": bool(checked.get("complete", False)),
                "covered_galaxy_ids": checked.get("covered_galaxy_ids"),
                "provenance": {"input_cache": str(directory.resolve()),
                               "merger_catalogs": str(catalog_path.resolve())}}

    analysis = galaxy_stripping_catalog(
        galaxy_histories, load_shock_product, centers,
        options=stripping_options, output_dir=output_dir, make_plots=make_plots)
    candidates = [row for row in analysis["classifications"]
                  if row["category"] == "merger_shock_stripping_candidate"]
    return {"analysis": analysis, "merger_catalogs": catalog_by_snapshot,
            "candidates": candidates,
            "candidate_galaxy_ids": [row["galaxy_id"] for row in candidates],
            "merger_catalog_path": str(catalog_path.resolve())}
