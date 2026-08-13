"""Build, inspect, save, and reload a ShockFinder surface catalog.

This example keeps the project's input convention: ``cell`` is an AMR cell
table extracted from part of one simulation snapshot. The extraction itself is
simulation-reader specific and should be performed before calling
``make_shock_catalog``.

Required fields are documented in the project README. In particular, positions
and cell widths are in km, velocities in km/s, temperature in K, and density in
Msol/kpc3. Extracted regions always use an open boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import shocktest


def make_shock_catalog(
    cell,
    output_dir,
    *,
    snapshot=None,
    region=None,
    minlevel=13,
    maxlevel=20,
    min_mach=1.5,
    gamma=5.0 / 3.0,
    mach_tolerance=0.3,
    normal_cosine=0.7,
    duplicate_normal_cosine=0.8,
    external_temperature=1.0e4,
    classification_fraction=0.8,
    show_progress=True,
    make_qa_plot=True,
):
    """Run the complete catalog workflow for one extracted AMR region.

    Returns
    -------
    analysis:
        ``ShockAnalysis`` containing the cell-level result, dissipation fields,
        catalog, timings, and counts. Call ``analysis.clear()`` after any
        cell-level products are no longer needed.
    paths:
        Paths of the versioned NPZ catalog, summary CSV, and optional QA image.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {}
    if snapshot is not None:
        provenance["snapshot"] = snapshot
    if region is not None:
        provenance["region"] = region

    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    finder.min_mach = min_mach
    finder.gamma = gamma
    finder.boundary = "open"
    finder.show_progress = show_progress

    analysis = finder.analyze(
        cell,
        compute_dissipation=True,
        build_catalog=True,
        deduplicate=True,
        mach_tolerance=mach_tolerance,
        normal_cosine=normal_cosine,
        duplicate_normal_cosine=duplicate_normal_cosine,
        min_mach=min_mach,
        external_temperature=external_temperature,
        classification_fraction=classification_fraction,
        provenance=provenance,
    )

    catalog_path = output_dir / "shock_catalog.npz"
    csv_path = output_dir / "shock_groups.csv"
    qa_path = output_dir / "shock_catalog_qa.png"

    shocktest.save_shock_catalog(catalog_path, analysis.catalog)
    shocktest.save_shock_catalog_csv(csv_path, analysis.catalog)

    summary = shocktest.summarize_catalog_quality(analysis.catalog)
    print("analysis counts:", analysis.counts)
    print("analysis timings [s]:", analysis.timings)
    print("catalog quality:", summary)
    print_group_preview(analysis.catalog)

    paths = {"catalog": catalog_path, "csv": csv_path, "qa": None}
    if make_qa_plot:
        figure, _ = shocktest.plot_catalog_quality(analysis.catalog)
        figure.savefig(qa_path, dpi=180)
        plt.close(figure)
        paths["qa"] = qa_path

    # Demonstrate that the complete catalog can be used without ``cell``.
    loaded = shocktest.load_shock_catalog(catalog_path)
    if loaded.metadata != analysis.catalog.metadata:
        raise RuntimeError("saved catalog metadata failed the round-trip check")

    return analysis, paths


def print_group_preview(catalog, *, limit=10):
    """Print a compact preview of the strongest deterministic group IDs."""

    print(f"shock groups: {len(catalog.groups)}")
    for group in catalog.groups[:limit]:
        flags = ",".join(group.quality_flags) or "none"
        print(
            f"group={group.group_id:5d} "
            f"M_peak={group.mach_peak:7.3f} "
            f"M_mean={group.mach_mean:7.3f} "
            f"area={group.area:.6e} {group.area_unit} "
            f"E_diss={group.dissipation_total:.6e} erg/s "
            f"class={group.classification:10s} "
            f"complete={group.is_complete!s:5s} "
            f"flags={flags}"
        )


def load_catalog_only(path):
    """Load and summarize a previously saved catalog without AMR cell data."""

    catalog = shocktest.load_shock_catalog(path)
    print("metadata:", catalog.metadata)
    print("quality:", shocktest.summarize_catalog_quality(catalog))
    print_group_preview(catalog)
    return catalog


# Typical usage after reading a region from a snapshot:
#
# analysis, paths = make_shock_catalog(
#     cell,
#     "output/shock_catalog_00620",
#     snapshot=620,
#     region="cluster-core",
#     minlevel=13,
#     maxlevel=20,
#     min_mach=1.5,
# )
#
# result = analysis.result
# dissipation = analysis.dissipation
# catalog = analysis.catalog
#
# # Use result/dissipation here for cell-level maps, then release large arrays.
# analysis.clear()
#
# # The compact catalog remains available from disk without the original cell.
# catalog = load_catalog_only(paths["catalog"])
