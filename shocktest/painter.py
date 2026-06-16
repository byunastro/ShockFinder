from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def rgb_image(
    data,
    *,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    log: bool = False,
    qscale: float = 1.0,
    bad_color=(0.0, 0.0, 0.0, 0.0),
):
    """Convert a scalar map to an RGBA image for ``ax.imshow``.

    If ``log=True``, ``vmin`` and ``vmax`` are interpreted in log10 units.
    ``qscale`` applies an asinh contrast stretch after normalization.
    """

    values = np.asarray(data, dtype=np.float64)
    plot_values = np.full(values.shape, np.nan, dtype=np.float64)
    if log:
        mask = values > 0.0
        plot_values[mask] = np.log10(values[mask])
    else:
        plot_values = values.copy()

    finite = np.isfinite(plot_values)
    if not np.any(finite):
        image = np.zeros(values.shape + (4,), dtype=np.float64)
        image[...] = bad_color
        return image

    lo = float(np.nanmin(plot_values) if vmin is None else vmin)
    hi = float(np.nanmax(plot_values) if vmax is None else vmax)
    if hi <= lo:
        hi = lo + 1.0

    normed = np.clip((plot_values - lo) / (hi - lo), 0.0, 1.0)
    if qscale and qscale > 1.0:
        normed = np.arcsinh(qscale * normed) / np.arcsinh(qscale)

    mapper = plt.get_cmap(cmap)
    image = mapper(normed)
    image[~finite] = bad_color
    return image


def show_map(
    ax,
    image,
    *,
    extent=None,
    origin: str = "lower",
    interpolation: str = "nearest",
    **imshow_kwargs,
):
    """Small wrapper around ``imshow`` for map images."""

    return ax.imshow(
        image,
        origin=origin,
        extent=extent,
        interpolation=interpolation,
        **imshow_kwargs,
    )


def plot_shock_maps(
    machmap,
    disspEmap,
    *,
    extent=None,
    cmap_diss="plasma",
    cmap_mach="RdYlBu_r",
    log_ediss_range: tuple[float, float] | None = None,
    log_mach_range: tuple[float, float] | None = None,
    qscale_diss: float = 1.0,
    qscale_mach: float = 1.0,
    titles=(r"Shock dissipation flux", r"Mach number"),
    circle_center: tuple[float, float] | None = None,
    circle_radius: float | None = None,
    circle_colors=("white", "black"),
    circle_kwargs: dict | None = None,
    scale_bar_length: float | None = None,
    scale_bar_label: str = "",
    scale_bar_colors=("white", "black"),
    figsize=(14, 6),
    output: str | Path | None = None,
    dpi: int = 200,
):
    """Draw left ``log E_diss`` and right ``log Mach`` maps.

    This function expects regular 2D arrays, for example the ``machmap`` and
    ``disspEmap`` returned by ``pyShockFinder.make_shock_maps``.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    diss_img = rgb_image(
        disspEmap,
        cmap=cmap_diss,
        log=True,
        vmin=None if log_ediss_range is None else log_ediss_range[0],
        vmax=None if log_ediss_range is None else log_ediss_range[1],
        qscale=qscale_diss,
    )
    mach_img = rgb_image(
        machmap,
        cmap=cmap_mach,
        log=True,
        vmin=None if log_mach_range is None else log_mach_range[0],
        vmax=None if log_mach_range is None else log_mach_range[1],
        qscale=qscale_mach,
    )

    show_map(axes[0], diss_img, extent=extent)
    show_map(axes[1], mach_img, extent=extent)
    for ax, title in zip(axes, titles):
        ax.set_aspect("equal")
        ax.set_title(title)
        if circle_center is not None and circle_radius is not None:
            kwargs = {"fill": False, "ls": ":", "lw": 1.5}
            if circle_kwargs:
                kwargs.update(circle_kwargs)
            ax.add_patch(Circle(circle_center, circle_radius, color=circle_colors[0 if ax is axes[0] else 1], **kwargs))
        if scale_bar_length is not None:
            color = scale_bar_colors[0 if ax is axes[0] else 1]
            draw_scale_bar(ax, scale_bar_length, scale_bar_label, color=color)

    add_colorbar(
        fig,
        axes[0],
        cmap=cmap_diss,
        vmin=None if log_ediss_range is None else log_ediss_range[0],
        vmax=None if log_ediss_range is None else log_ediss_range[1],
        label=r"$\log E_{\rm diss}\ [{\rm erg\ s^{-1}\ kpc^{-2}}]$",
    )
    add_colorbar(
        fig,
        axes[1],
        cmap=cmap_mach,
        vmin=None if log_mach_range is None else log_mach_range[0],
        vmax=None if log_mach_range is None else log_mach_range[1],
        label=r"$\log \mathcal{M}$",
    )

    if output is not None:
        fig.savefig(output, dpi=dpi)
    return fig, axes


def add_colorbar(fig, ax, *, cmap, vmin, vmax, label):
    if vmin is None or vmax is None:
        return None
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.08)
    cbar.set_label(label)
    return cbar


def draw_scale_bar(ax, length: float, label: str, *, color="white", lw: float = 4.0):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmax - 0.08 * (xmax - xmin) - length
    x1 = x0 + length
    y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot([x0, x1], [y0, y0], color=color, lw=lw, solid_capstyle="butt")
    if label:
        ax.text(
            0.5 * (x0 + x1),
            y0 + 0.025 * (ymax - ymin),
            label,
            color=color,
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )
