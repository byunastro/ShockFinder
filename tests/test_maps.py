import numpy as np
import pytest

from shocktest import painter, pyShockFinder


def grid_cell(nx=10, ny=4):
    x0, y0 = np.meshgrid(np.arange(nx, dtype=float) + 0.5, np.arange(ny, dtype=float) + 0.5)
    x = x0.ravel()
    y = y0.ravel()
    n = x.size
    temp = np.full(n, 1.0e7)
    rho = np.full(n, 1.0e6)
    vx = np.ones(n)
    right = x >= nx / 2
    temp[right] = 4.0e7
    rho[right] = 3.0e6
    vx[right] = -1.0
    return {
        ("x", "km"): x,
        ("y", "km"): y,
        ("z", "km"): np.full(n, 0.5),
        ("dx", "km"): np.ones(n),
        ("vx", "km/s"): vx,
        ("vy", "km/s"): np.zeros(n),
        ("vz", "km/s"): np.zeros(n),
        ("T", "K"): temp,
        ("rho", "Msol/kpc3"): rho,
        "level": np.full(n, 20, dtype=np.int32),
    }


def test_make_maps_and_rgb_image():
    maps = pyShockFinder.make_shock_maps(
        grid_cell(),
        minlevel=13,
        maxlevel=20,
        bins=(16, 16),
        show_progress=False,
    )

    assert maps.machmap.shape == (16, 16)
    assert maps.disspEmap.shape == (16, 16)
    assert np.nanmax(maps.machmap) > 1.0
    assert np.nanmax(maps.disspEmap) > 0.0
    assert maps.dissipation.area.shape == maps.dissipation.flux.shape

    image = painter.rgb_image(maps.disspEmap, cmap="plasma", log=True, qscale=4.0)
    assert image.shape == (16, 16, 4)


def test_make_mach_map_from_result_like_user_example():
    import shocktest

    finder = shocktest.ShockFinder()
    finder.minlevel = 15
    finder.maxlevel = 20
    finder.show_progress = False

    result = finder.ShockFinder(grid_cell())
    machmap = painter.make_mach_map(result, plane="xy", statistic="mean", bins=(16, 16))

    assert machmap.shape == (16, 16)
    assert np.nanmax(np.log10(machmap)) > 0.0


def test_make_mach_map_accepts_dissipation_weights():
    import shocktest

    result = shocktest.ShockResult(
        mach=np.array([2.0, 10.0]),
        shock=np.array([True, True]),
        center_index=np.array([0, 1]),
        upstream_index=np.array([0, 1]),
        downstream_index=np.array([0, 1]),
        selected_indices=np.array([0, 1]),
        pos=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        dx=np.array([1.0, 1.0]),
    )
    weights = np.array([9.0, 1.0])

    machmap = painter.make_mach_map(
        result,
        plane="xy",
        bins=(1, 1),
        extent=(0.0, 1.0, 0.0, 1.0),
        statistic="mean",
        method="amr",
        weights=weights,
    )

    np.testing.assert_allclose(machmap, np.array([[2.8]]))
    with pytest.raises(ValueError, match="weights"):
        painter.make_mach_map(result, statistic="max", weights=weights)


def test_maps_use_both_shock_and_validated_mach_masks():
    import shocktest

    result = shocktest.ShockResult(
        mach=np.array([2.0, 3.0, 4.0]),
        shock=np.array([True, False, True]),
        center_index=np.arange(3),
        upstream_index=np.arange(3),
        downstream_index=np.arange(3),
        selected_indices=np.arange(3),
        pos=np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]),
        dx=np.ones(3),
        mach_consistent=np.array([True, True, False]),
    )
    dissipation = pyShockFinder.DissipationResult(
        flux=np.array([10.0, 20.0, 30.0]),
        total=np.zeros(3),
        area=np.ones(3),
        efficiency=np.ones(3),
        sound_speed=np.ones(3),
    )
    kwargs = {"bins": (1, 3), "extent": (0.0, 3.0, 0.0, 1.0)}

    machmap = painter.make_mach_map(result, **kwargs)
    dissp_map = painter.make_disspE_map(result, dissipation, **kwargs)

    np.testing.assert_allclose(machmap, np.array([[2.0, np.nan, np.nan]]), equal_nan=True)
    np.testing.assert_allclose(dissp_map, np.array([[10.0, np.nan, np.nan]]), equal_nan=True)

    override = painter.make_mach_map(
        result,
        valid_mach=np.array([False, False, True]),
        **kwargs,
    )
    np.testing.assert_allclose(override, np.array([[np.nan, np.nan, 4.0]]), equal_nan=True)


def test_fill_small_map_gaps_uses_log_harmonic_interpolation():
    raw = np.array(
        [
            [4.0, 4.0, 4.0],
            [np.nan, np.nan, np.nan],
            [16.0, 16.0, 16.0],
        ]
    )

    filled, fill_mask = painter.fill_small_map_gaps(raw, return_mask=True)

    np.testing.assert_allclose(filled[1], np.full(3, 8.0), rtol=1.0e-5)
    np.testing.assert_array_equal(fill_mask[1], np.ones(3, dtype=bool))
    np.testing.assert_array_equal(filled[[0, 2]], raw[[0, 2]])


def test_fill_small_map_gaps_does_not_bridge_wide_or_unbounded_regions():
    raw = np.array(
        [
            [4.0, np.nan, np.nan, np.nan, 16.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    filled, fill_mask = painter.fill_small_map_gaps(raw, return_mask=True)

    np.testing.assert_allclose(filled, raw, equal_nan=True)
    assert not np.any(fill_mask)


def test_area_painting_fills_projected_cell_footprint():
    x = np.array([0.5])
    y = np.array([0.5])
    dx = np.array([1.0])
    values = np.array([3.0])

    area_map = painter._paint_cells_to_map(x, y, dx, values, bins=(2, 2), extent=(0.0, 1.0, 0.0, 1.0), statistic="mean")
    point_map = painter._bin_to_map(x, y, values, bins=(2, 2), extent=(0.0, 1.0, 0.0, 1.0), statistic="mean")

    np.testing.assert_allclose(area_map, np.full((2, 2), 3.0))
    assert np.count_nonzero(np.isfinite(point_map)) == 1


def test_area_sum_scales_by_pixel_coverage():
    x = np.array([0.25])
    y = np.array([0.25])
    dx = np.array([0.5])
    values = np.array([8.0])

    area_map = painter._paint_cells_to_map(x, y, dx, values, bins=(1, 1), extent=(0.0, 1.0, 0.0, 1.0), statistic="sum")

    np.testing.assert_allclose(area_map, np.array([[2.0]]))


def test_amr_auto_backend_falls_back_when_fortran_kernel_is_unavailable(monkeypatch):
    x = np.array([0.25, 0.75])
    y = np.array([0.25, 0.75])
    dx = np.array([0.5, 0.5])
    values = np.array([8.0, 4.0])
    kwargs = {"bins": (2, 2), "extent": (0.0, 1.0, 0.0, 1.0)}
    monkeypatch.setattr(painter, "_FORTRAN_MAP_KERNEL", None)

    automatic = painter._paint_cells_to_map(
        x, y, dx, values, statistic="mean", backend="auto", **kwargs
    )
    reference = painter._paint_cells_to_map(
        x, y, dx, values, statistic="mean", backend="python", **kwargs
    )

    np.testing.assert_allclose(automatic, reference, equal_nan=True)
    with pytest.raises(ImportError, match="rebuild"):
        painter._paint_cells_to_map(
            x, y, dx, values, statistic="mean", backend="fortran", **kwargs
        )


@pytest.mark.parametrize("statistic", ["max", "mean", "sum"])
@pytest.mark.parametrize("weighted", [False, True])
def test_amr_fortran_backend_matches_python(statistic, weighted):
    if painter._FORTRAN_MAP_KERNEL is None:
        pytest.skip("Fortran map rasterizer is not built")

    x = np.array([-0.1, 0.15, 0.5, 0.85, 1.1, np.nan])
    y = np.array([0.5, 0.15, 0.5, 0.85, 0.5, 0.5])
    dx = np.array([0.4, 0.3, 0.6, 0.3, 0.4, 0.2])
    values = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    weights = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 1.0]) if weighted else None
    kwargs = {
        "bins": (7, 9),
        "extent": (0.0, 1.0, 0.0, 1.0),
        "statistic": statistic,
        "weights": weights,
    }

    expected = painter._paint_cells_to_map(
        x, y, dx, values, backend="python", **kwargs
    )
    actual = painter._paint_cells_to_map(
        x, y, dx, values, backend="fortran", **kwargs
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, equal_nan=True)


def test_normal_shock_surface_area_matches_cell_area_for_axis_aligned_shock():
    import shocktest

    finder = shocktest.ShockFinder()
    finder.minlevel = 15
    finder.maxlevel = 20
    finder.show_progress = False

    cell = grid_cell()
    result = finder.ShockFinder(cell)
    diss_cell = pyShockFinder.compute_dissipation(cell, result, area_mode="cell")
    diss_normal = pyShockFinder.compute_dissipation(cell, result, area_mode="normal")

    valid = diss_cell.area > 0.0
    np.testing.assert_allclose(diss_normal.area[valid], diss_cell.area[valid])
    np.testing.assert_allclose(diss_normal.total[valid], diss_cell.total[valid])


def test_dissipation_uses_upstream_temperature_floor():
    import shocktest

    cell = {
        ("T", "K"): np.array([1.0, 2.0e4]),
        ("rho", "Msol/kpc3"): np.array([1.0e6, 2.0e6]),
        ("dx", "km"): np.array([1.0, 1.0]),
    }
    result = shocktest.ShockResult(
        mach=np.array([0.0, 2.0]),
        shock=np.array([False, True]),
        center_index=np.array([-1, 1]),
        upstream_index=np.array([-1, 0]),
        downstream_index=np.array([-1, 1]),
        selected_indices=np.array([0, 1]),
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        dx=np.array([1.0, 1.0]),
    )

    diss = pyShockFinder.compute_dissipation(cell, result, temperature_floor=1.0e4)

    assert np.isfinite(diss.sound_speed[1])
    assert diss.sound_speed[1] > 0.0
    assert diss.flux[1] > 0.0


def test_shock_map_result_clear_releases_nested_arrays():
    maps = pyShockFinder.make_shock_maps(
        grid_cell(),
        minlevel=13,
        maxlevel=20,
        bins=(8, 8),
        show_progress=False,
    )

    maps.clear()

    assert maps.machmap.shape == (0, 0)
    assert maps.disspEmap.shape == (0, 0)
    assert maps.result is None
    assert maps.dissipation is None
