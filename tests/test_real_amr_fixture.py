from pathlib import Path

import numpy as np
import pytest

import shocktest


FIXTURE = Path(__file__).resolve().parents[1] / "amr_data"
KPC_IN_KM = 3.0856775814913673e16


@pytest.mark.skipif(not (FIXTURE / "metadata.json").exists(), reason="local AMR fixture absent")
def test_real_amr_center_subbox_regression():
    def load(name):
        return np.load(FIXTURE / f"{name}.npy", mmap_mode="r", allow_pickle=False)

    x = load("x_km")
    y = load("y_km")
    z = load("z_km")
    center = np.array(
        [
            0.5 * (float(np.min(x)) + float(np.max(x))),
            0.5 * (float(np.min(y)) + float(np.max(y))),
            0.5 * (float(np.min(z)) + float(np.max(z))),
        ]
    )
    half_width = 12.5 * KPC_IN_KM
    rows = np.nonzero(
        (np.abs(x - center[0]) <= half_width)
        & (np.abs(y - center[1]) <= half_width)
        & (np.abs(z - center[2]) <= half_width)
    )[0]

    def take(name):
        return np.asarray(load(name)[rows])

    cell = {
        ("x", "km"): np.asarray(x[rows]),
        ("y", "km"): np.asarray(y[rows]),
        ("z", "km"): np.asarray(z[rows]),
        ("dx", "km"): take("dx_km"),
        ("vx", "km/s"): take("vx_kms"),
        ("vy", "km/s"): take("vy_kms"),
        ("vz", "km/s"): take("vz_kms"),
        ("T", "K"): take("temperature_K"),
        ("rho", "Msol/kpc3"): take("density_Msol_kpc3"),
        "level": take("level"),
    }

    finder = shocktest.ShockFinder()
    result = finder.find(cell)

    assert rows.size == 1_500_820
    assert np.count_nonzero(result.shock) == 107_578
    assert np.all(np.isfinite(result.mach[result.shock]))
    assert result.diagnostics["center_step_limit"] == 0
