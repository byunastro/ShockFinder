from __future__ import annotations

# Based on Ryu et al. 2003, ApJ, 593, 599.

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

import shocktest
from shocktest import painter, pyShockFinder

from synthetic_shocks import (
    external_internal_masks,
    multi_shock_line_cell,
    planar_shock_cell,
    shock_surface_histogram,
    tiled_sheet_cell,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    line_cell = tiled_sheet_cell(multi_shock_line_cell(), ny=24)
    line_result = finder.find(line_cell)
    external, internal = external_internal_masks(line_cell, line_result)

    bins = np.array([1.5, 2.0, 3.0, 5.0, 10.0])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for mask, label in [(external, "External"), (internal, "Internal")]:
        subset = line_result
        original_shock = subset.shock.copy()
        subset.shock = mask
        surface = shock_surface_histogram(subset, bins=bins)
        subset.shock = original_shock
        centers = np.sqrt(bins[:-1] * bins[1:])
        ax.step(centers, surface, where="mid", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Surface proxy dS/dlogM")
    ax.legend()
    fig.savefig(output_dir / "synthetic_shock_surface_histogram.png", dpi=200)
    plt.close(fig)

    sheet_cell = tiled_sheet_cell(planar_shock_cell(3.0, n=48, shock_index=24), ny=32)
    sheet_result = finder.find(sheet_cell)
    diss = pyShockFinder.compute_dissipation(sheet_cell, sheet_result)
    maps = pyShockFinder.make_shock_maps(
        sheet_cell,
        result=sheet_result,
        dissipation=diss,
        minlevel=0,
        maxlevel=20,
        plane="xy",
        bins=(96, 64),
        min_mach=1.5,
        show_progress=False,
    )
    fig, _ = painter.plot_shock_maps(
        maps.machmap,
        maps.disspEmap,
        extent=maps.extent,
        log_mach_range=(0.0, 0.7),
        output=output_dir / "synthetic_shock_maps.png",
    )
    plt.close(fig)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
