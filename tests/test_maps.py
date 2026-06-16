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
