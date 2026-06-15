from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import shocktest


def compute_shock_result(cell, minlevel: int = 13, maxlevel: int = 20):
    finder = shocktest.ShockFinder()
    finder.minlevel = minlevel
    finder.maxlevel = maxlevel
    return finder.ShockFinder(cell)


def plot_mach_number_map(
    cell,
    result,
    *,
    plane: str = "xy",
    z_center: float | None = None,
    z_width: float | None = None,
    min_mach: float = 1.0,
    output: str | Path | None = "mach_map.png",
):
    """Draw an AMR-aware 2D Mach number map from a ShockFinder result.

    Parameters
    ----------
    cell:
        AMR cell table with tuple-key fields, e.g. ``cell['x', 'km']``.
    result:
        Output from ``shocktest.ShockFinder().ShockFinder(cell)``.
    plane:
        Projection plane: ``"xy"``, ``"xz"``, or ``"yz"``.
    z_center, z_width:
        Optional slab selection along the axis normal to ``plane``.
    min_mach:
        Minimum Mach number to draw.
    output:
        Output PNG path. Use ``None`` to show interactively only.
    """

    axes = {
        "xy": (("x", "km"), ("y", "km"), ("z", "km")),
        "xz": (("x", "km"), ("z", "km"), ("y", "km")),
        "yz": (("y", "km"), ("z", "km"), ("x", "km")),
    }
    if plane not in axes:
        raise ValueError("plane must be one of: xy, xz, yz")

    x_key, y_key, normal_key = axes[plane]
    original_rows = result.selected_indices
    x = np.asarray(cell[x_key])[original_rows]
    y = np.asarray(cell[y_key])[original_rows]
    normal = np.asarray(cell[normal_key])[original_rows]
    dx = np.asarray(cell["dx", "km"])[original_rows]

    draw_mask = result.shock & (result.mach >= min_mach)
    if z_center is not None and z_width is not None:
        half_width = 0.5 * z_width
        draw_mask &= np.abs(normal - z_center) <= half_width

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.set_aspect("equal")

    if not np.any(draw_mask):
        ax.text(0.5, 0.5, "No shock cells in selected slice", transform=ax.transAxes, ha="center")
        ax.set_xlabel(f"{x_key[0]} [{x_key[1]}]")
        ax.set_ylabel(f"{y_key[0]} [{y_key[1]}]")
        if output is not None:
            fig.savefig(output, dpi=200)
        return fig, ax

    patches = [
        Rectangle((x_i - 0.5 * dx_i, y_i - 0.5 * dx_i), dx_i, dx_i)
        for x_i, y_i, dx_i in zip(x[draw_mask], y[draw_mask], dx[draw_mask])
    ]
    colors = result.mach[draw_mask]
    collection = PatchCollection(patches, array=colors, cmap="magma", edgecolor="none")
    collection.set_clim(vmin=max(min_mach, float(np.nanmin(colors))), vmax=float(np.nanmax(colors)))
    ax.add_collection(collection)

    pad = 0.5 * np.max(dx[draw_mask])
    ax.set_xlim(np.min(x[draw_mask] - 0.5 * dx[draw_mask]) - pad, np.max(x[draw_mask] + 0.5 * dx[draw_mask]) + pad)
    ax.set_ylim(np.min(y[draw_mask] - 0.5 * dx[draw_mask]) - pad, np.max(y[draw_mask] + 0.5 * dx[draw_mask]) + pad)
    ax.set_xlabel(f"{x_key[0]} [{x_key[1]}]")
    ax.set_ylabel(f"{y_key[0]} [{y_key[1]}]")
    ax.set_title(f"Shock Mach number map ({plane})")

    cbar = fig.colorbar(collection, ax=ax)
    cbar.set_label("Mach number")

    if output is not None:
        fig.savefig(output, dpi=200)
    return fig, ax


# Example use after your simulation code has created `cell`:
#
# from examples.plot_mach_map import compute_shock_result, plot_mach_number_map
#
# result = compute_shock_result(cell, minlevel=13, maxlevel=20)
# plot_mach_number_map(
#     cell,
#     result,
#     plane="xy",
#     z_center=None,
#     z_width=None,
#     min_mach=1.0,
#     output="mach_map_xy.png",
# )
# plt.show()
