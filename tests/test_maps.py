import numpy as np

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
