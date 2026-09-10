from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


Plane = Literal["xy", "xz", "yz"]
Statistic = Literal["max", "sum", "mean"]


def make_mach_map(
    result,
    *,
    plane: Plane = "xy",
    bins: int | tuple[int, int] = 1024,
    extent: tuple[float, float, float, float] | None = None,
    z_center: float | None = None,
    z_width: float | None = None,
    min_mach: float = 1.0,
    statistic: Statistic = "max",
    method: Literal["amr", "point"] = "amr",
    weights=None,
    valid_mach=None,
    fill_gaps: int = 0,
):
    """Make a regular 2D Mach map from a ``ShockResult``.

    This supports the workflow:
    ``machmap = painter.make_mach_map(result, plane="xy")``.
    The default ``method="amr"`` paints each projected AMR cell footprint into
    the image. ``method="point"`` keeps the older center-binning behavior.
    By default, ``result.mach_consistent`` is used as the validation mask when
    it is available. Pass ``valid_mach`` explicitly to override that mask.
    Empty pixels are returned as NaN. Pass per-cell ``weights`` with
    ``statistic="mean"`` to make an area- and weight-averaged Mach map, for
    example using shock dissipation energy as the weight. ``fill_gaps`` can be
    set to a small positive pixel width (normally 1) to inpaint only narrow,
    bracketed display gaps in log space. It is disabled by default so the raw
    map remains suitable for quantitative work.
    """

    x, y, normal, dx = _project_result_geometry(result, plane)
    if extent is None:
        extent = map_extent_from_result(result, plane=plane)

    draw = _shock_draw_mask(result, min_mach=min_mach, valid_mach=valid_mach)
    map_weights = None
    if weights is not None:
        if statistic == "max":
            raise ValueError('weights are only meaningful for statistic="mean" or statistic="sum"')
        map_weights = np.asarray(weights, dtype=np.float64)
        if map_weights.shape != result.mach.shape:
            raise ValueError("weights must have the same shape as result.mach")
    if z_center is not None and z_width is not None:
        draw &= np.abs(normal - z_center) <= 0.5 * z_width
    if map_weights is not None:
        map_weights = map_weights[draw]
    mapped = _values_to_map(
        x[draw],
        y[draw],
        dx[draw],
        result.mach[draw],
        bins,
        extent,
        statistic,
        method,
        weights=map_weights,
    )
    if fill_gaps:
        mapped = fill_small_map_gaps(mapped, max_gap_pixels=fill_gaps, log=True)
    return mapped


def make_disspE_map(
    result,
    dissipation,
    *,
    plane: Plane = "xy",
    bins: int | tuple[int, int] = 1024,
    extent: tuple[float, float, float, float] | None = None,
    z_center: float | None = None,
    z_width: float | None = None,
    min_mach: float = 1.0,
    statistic: Statistic = "max",
    method: Literal["amr", "point"] = "amr",
    valid_mach=None,
    fill_gaps: int = 0,
):
    """Make a regular 2D dissipation-flux map from a ``ShockResult``.

    The returned map is E_diss/A in ``erg s^-1 kpc^-2`` when ``dissipation`` is
    produced by ``pyShockFinder.compute_dissipation``. The default
    ``method="amr"`` paints each projected AMR shock-cell footprint into the
    image. ``method="point"`` keeps the older center-binning behavior. By
    default, ``result.mach_consistent`` is used as the validation mask when it
    exists; pass ``valid_mach`` explicitly to override it. ``fill_gaps`` has
    the same conservative, display-only meaning as in :func:`make_mach_map`.
    """

    x, y, normal, dx = _project_result_geometry(result, plane)
    if extent is None:
        extent = map_extent_from_result(result, plane=plane)

    draw = _shock_draw_mask(result, min_mach=min_mach, valid_mach=valid_mach)
    draw &= np.isfinite(dissipation.flux) & (dissipation.flux > 0.0)
    if z_center is not None and z_width is not None:
        draw &= np.abs(normal - z_center) <= 0.5 * z_width
    mapped = _values_to_map(
        x[draw],
        y[draw],
        dx[draw],
        dissipation.flux[draw],
        bins,
        extent,
        statistic,
        method,
    )
    if fill_gaps:
        mapped = fill_small_map_gaps(mapped, max_gap_pixels=fill_gaps, log=True)
    return mapped


def make_disspE_maps(*args, **kwargs):
    """Backward-compatible alias for ``make_disspE_map``."""

    return make_disspE_map(*args, **kwargs)


def map_extent_from_result(result, *, plane: Plane = "xy"):
    x, y, _, dx = _project_result_geometry(result, plane)
    if x.size == 0:
        return (0.0, 1.0, 0.0, 1.0)
    return (
        float(np.min(x - 0.5 * dx)),
        float(np.max(x + 0.5 * dx)),
        float(np.min(y - 0.5 * dx)),
        float(np.max(y + 0.5 * dx)),
    )


def fill_small_map_gaps(
    data,
    *,
    max_gap_pixels: int = 1,
    log: bool = True,
    max_iterations: int = 200,
    tolerance: float = 1.0e-6,
    return_mask: bool = False,
):
    """Inpaint narrow, bracketed gaps without expanding the map boundary.

    A missing pixel is eligible only when finite pixels bracket it along a
    horizontal, vertical, or diagonal direction within ``max_gap_pixels``.
    Eligible values are the discrete harmonic continuation of their eight
    neighboring pixels. Positive maps should use the default ``log=True`` so
    interpolation follows their multiplicative dynamic range.

    This is intended for visualization, not for recomputing integrated shock
    area or dissipated energy. ``return_mask=True`` also returns the pixels
    actually filled, which should be retained for provenance and QA plots.
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D map")
    if not isinstance(max_gap_pixels, (int, np.integer)) or max_gap_pixels < 0:
        raise ValueError("max_gap_pixels must be a non-negative integer")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    out = values.copy()
    empty_mask = np.zeros(values.shape, dtype=bool)
    if max_gap_pixels == 0 or values.size == 0:
        return (out, empty_mask) if return_mask else out

    valid = np.isfinite(values)
    if log:
        valid &= values > 0.0
    target = _bracketed_gap_mask(valid, int(max_gap_pixels))
    if not np.any(target):
        return (out, empty_mask) if return_mask else out

    work = np.full(values.shape, np.nan, dtype=np.float64)
    work[valid] = np.log(values[valid]) if log else values[valid]

    for _ in range(int(max_iterations)):
        neighbor_sum = np.zeros(values.shape, dtype=np.float64)
        neighbor_count = np.zeros(values.shape, dtype=np.int16)
        for dy, dx in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            neighbor = _shift_2d(work, dy, dx, np.nan)
            finite_neighbor = np.isfinite(neighbor)
            neighbor_sum[finite_neighbor] += neighbor[finite_neighbor]
            neighbor_count[finite_neighbor] += 1

        update = target & (neighbor_count > 0)
        if not np.any(update):
            break
        next_work = work.copy()
        next_work[update] = neighbor_sum[update] / neighbor_count[update]

        comparable = update & np.isfinite(work)
        converged = np.all(np.isfinite(next_work[target]))
        if converged and np.any(comparable):
            converged &= np.max(np.abs(next_work[comparable] - work[comparable])) <= tolerance
        elif converged:
            converged = False
        work = next_work
        if converged:
            break

    filled = target & np.isfinite(work)
    out[filled] = np.exp(work[filled]) if log else work[filled]
    return (out, filled) if return_mask else out


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
    interpolation: str = "none",
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
    """Draw left ``log E_diss`` and right ``log Mach`` maps."""

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
    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_aspect("equal")
        ax.set_title(title)
        if circle_center is not None and circle_radius is not None:
            kwargs = {"fill": False, "ls": ":", "lw": 1.5}
            if circle_kwargs:
                kwargs.update(circle_kwargs)
            ax.add_patch(Circle(circle_center, circle_radius, color=circle_colors[idx], **kwargs))
        if scale_bar_length is not None:
            draw_scale_bar(ax, scale_bar_length, scale_bar_label, color=scale_bar_colors[idx])

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


def _project_result_geometry(result, plane: Plane):
    if result.pos is None or result.dx is None:
        raise ValueError("ShockResult does not contain geometry. Re-run ShockFinder with the updated code.")
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    if plane not in axes:
        raise ValueError("plane must be one of: xy, xz, yz")
    x_axis, y_axis, normal_axis = axes[plane]
    pos = np.asarray(result.pos, dtype=np.float64)
    dx = np.asarray(result.dx, dtype=np.float64)
    return pos[:, x_axis], pos[:, y_axis], pos[:, normal_axis], dx


def _shock_draw_mask(result, *, min_mach: float, valid_mach=None):
    mach = np.asarray(result.mach, dtype=np.float64)
    shock = np.asarray(result.shock, dtype=bool)
    if shock.shape != mach.shape:
        raise ValueError("result.shock must have the same shape as result.mach")

    if valid_mach is None:
        valid_mach = getattr(result, "mach_consistent", None)
    if valid_mach is None:
        valid = np.ones(mach.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mach, dtype=bool)
        if valid.shape != mach.shape:
            raise ValueError("valid_mach must have the same shape as result.mach")
    return shock & valid & np.isfinite(mach) & (mach >= min_mach)


def _bracketed_gap_mask(valid, max_gap_pixels: int):
    missing = ~valid
    bracketed = np.zeros(valid.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        negative = np.zeros(valid.shape, dtype=bool)
        positive = np.zeros(valid.shape, dtype=bool)
        for distance in range(1, max_gap_pixels + 1):
            negative |= _shift_2d(valid, -dy * distance, -dx * distance, False)
            positive |= _shift_2d(valid, dy * distance, dx * distance, False)
        bracketed |= negative & positive
    return missing & bracketed


def _shift_2d(values, dy: int, dx: int, fill_value):
    shifted = np.full(values.shape, fill_value, dtype=values.dtype)
    ny, nx = values.shape
    src_y = slice(max(0, -dy), min(ny, ny - dy))
    src_x = slice(max(0, -dx), min(nx, nx - dx))
    dst_y = slice(max(0, dy), min(ny, ny + dy))
    dst_x = slice(max(0, dx), min(nx, nx + dx))
    shifted[dst_y, dst_x] = values[src_y, src_x]
    return shifted


def _bin_shape(bins: int | tuple[int, int]):
    if isinstance(bins, int):
        return bins, bins
    if len(bins) != 2:
        raise ValueError("bins must be an int or (ny, nx)")
    return int(bins[0]), int(bins[1])


def _values_to_map(x, y, dx, values, bins, extent, statistic: Statistic, method: Literal["amr", "point"], *, weights=None):
    if method == "point":
        return _bin_to_map(x, y, values, bins, extent, statistic, weights=weights)
    if method == "amr":
        return _paint_cells_to_map(x, y, dx, values, bins, extent, statistic, weights=weights)
    raise ValueError("method must be one of: amr, point")


def _bin_to_map(x, y, values, bins, extent, statistic: Statistic, *, weights=None):
    ny, nx = _bin_shape(bins)
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = _optional_weights(weights, values.size)
    if values.size == 0:
        return out

    xmin, xmax, ymin, ymax = extent
    ix = np.floor((x - xmin) / (xmax - xmin) * nx).astype(np.int64)
    iy = np.floor((y - ymin) / (ymax - ymin) * ny).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(values)
    if weights is not None:
        inside &= np.isfinite(weights) & (weights > 0.0)
    if not np.any(inside):
        return out

    flat = iy[inside] * nx + ix[inside]
    vals = values[inside]
    if statistic == "max":
        work = np.full(ny * nx, -np.inf, dtype=np.float64)
        np.maximum.at(work, flat, vals)
        work[~np.isfinite(work)] = np.nan
        return work.reshape(ny, nx)
    if statistic == "sum":
        work = np.zeros(ny * nx, dtype=np.float64)
        if weights is not None:
            vals = vals * weights[inside]
        np.add.at(work, flat, vals)
        work[work == 0.0] = np.nan
        return work.reshape(ny, nx)
    if statistic == "mean":
        work = np.zeros(ny * nx, dtype=np.float64)
        if weights is None:
            count = np.zeros(ny * nx, dtype=np.int64)
            np.add.at(work, flat, vals)
            np.add.at(count, flat, 1)
            valid = count > 0
            work[valid] /= count[valid]
        else:
            weight_sum = np.zeros(ny * nx, dtype=np.float64)
            vals_weight = weights[inside]
            np.add.at(work, flat, vals * vals_weight)
            np.add.at(weight_sum, flat, vals_weight)
            valid = weight_sum > 0.0
            work[valid] /= weight_sum[valid]
        work[~valid] = np.nan
        return work.reshape(ny, nx)
    raise ValueError("statistic must be one of: max, sum, mean")


def _paint_cells_to_map(x, y, dx, values, bins, extent, statistic: Statistic, *, weights=None):
    ny, nx = _bin_shape(bins)
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = _optional_weights(weights, values.size)
    if values.size == 0:
        return out

    xmin, xmax, ymin, ymax = extent
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("extent must be (xmin, xmax, ymin, ymax) with positive width and height")

    pixw = (xmax - xmin) / nx
    pixh = (ymax - ymin) / ny
    if pixw <= 0.0 or pixh <= 0.0:
        raise ValueError("bins and extent produce non-positive pixel size")

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & (dx > 0.0) & np.isfinite(values)
    if weights is not None:
        finite &= np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return out
    x = x[finite]
    y = y[finite]
    dx = dx[finite]
    values = values[finite]
    if weights is not None:
        weights = weights[finite]

    if statistic == "max":
        work = np.full((ny, nx), -np.inf, dtype=np.float64)
        for xc, yc, width, value in zip(x, y, dx, values):
            ix0, ix1, iy0, iy1 = _cell_pixel_bounds(xc, yc, width, xmin, ymin, pixw, pixh, nx, ny)
            if ix0 > ix1 or iy0 > iy1:
                continue
            np.maximum(work[iy0 : iy1 + 1, ix0 : ix1 + 1], value, out=work[iy0 : iy1 + 1, ix0 : ix1 + 1])
        work[~np.isfinite(work)] = np.nan
        return work

    if statistic in {"mean", "sum"}:
        value_sum = np.zeros((ny, nx), dtype=np.float64)
        area_sum = np.zeros((ny, nx), dtype=np.float64)
        pixel_area = pixw * pixh
        pixel_x0 = xmin + np.arange(nx, dtype=np.float64) * pixw
        pixel_y0 = ymin + np.arange(ny, dtype=np.float64) * pixh

        if weights is None:
            iterator = zip(x, y, dx, values, np.ones(values.size, dtype=np.float64))
        else:
            iterator = zip(x, y, dx, values, weights)

        for xc, yc, width, value, weight in iterator:
            ix0, ix1, iy0, iy1 = _cell_pixel_bounds(xc, yc, width, xmin, ymin, pixw, pixh, nx, ny)
            if ix0 > ix1 or iy0 > iy1:
                continue

            left = xc - 0.5 * width
            right = xc + 0.5 * width
            bottom = yc - 0.5 * width
            top = yc + 0.5 * width
            px0 = pixel_x0[ix0 : ix1 + 1]
            py0 = pixel_y0[iy0 : iy1 + 1]
            ox = np.maximum(0.0, np.minimum(right, px0 + pixw) - np.maximum(left, px0))
            oy = np.maximum(0.0, np.minimum(top, py0 + pixh) - np.maximum(bottom, py0))
            if not np.any(ox > 0.0) or not np.any(oy > 0.0):
                continue

            overlap = oy[:, None] * ox[None, :]
            target = (slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))
            weighted_overlap = weight * overlap
            value_sum[target] += value * weighted_overlap
            area_sum[target] += weighted_overlap

        valid = area_sum > 0.0
        if statistic == "mean":
            out[valid] = value_sum[valid] / area_sum[valid]
        else:
            out[valid] = value_sum[valid] / pixel_area
        return out

    raise ValueError("statistic must be one of: max, sum, mean")


def _optional_weights(weights, size: int):
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (size,):
        raise ValueError("weights must be a 1D array with the same length as values")
    return weights


def _cell_pixel_bounds(x, y, dx, xmin, ymin, pixw, pixh, nx, ny):
    left = x - 0.5 * dx
    right = x + 0.5 * dx
    bottom = y - 0.5 * dx
    top = y + 0.5 * dx

    ix0 = max(0, int(np.floor((left - xmin) / pixw)))
    ix1 = min(nx - 1, int(np.ceil((right - xmin) / pixw) - 1))
    iy0 = max(0, int(np.floor((bottom - ymin) / pixh)))
    iy1 = min(ny - 1, int(np.ceil((top - ymin) / pixh) - 1))
    return ix0, ix1, iy0, iy1
