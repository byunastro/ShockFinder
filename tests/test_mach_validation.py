from __future__ import annotations

import numpy as np
import pytest

import shocktest
from shocktest.mach_validation import (
    MachValidationFlag,
    jump_ratio,
    mach_from_density_jump,
    mach_from_density_ratio,
    mach_from_pressure_jump,
    mach_from_pressure_ratio,
    mach_from_temperature_jump,
    mach_from_temperature_ratio,
)

from synthetic_shocks import (
    density_jump_from_mach,
    planar_shock_cell,
    temperature_jump_from_mach,
)


GAMMA = 5.0 / 3.0
MACH_VALUES = np.array([1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0])


def pressure_jump_from_mach(mach, gamma=GAMMA):
    m2 = np.asarray(mach, dtype=np.float64) ** 2
    return (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)


def test_analytic_jump_estimators_recover_input_mach():
    temperature_ratio = np.array(
        [temperature_jump_from_mach(value) for value in MACH_VALUES]
    )
    pressure_ratio = pressure_jump_from_mach(MACH_VALUES)
    density_ratio = np.array(
        [density_jump_from_mach(value) for value in MACH_VALUES]
    )

    np.testing.assert_allclose(
        mach_from_temperature_ratio(temperature_ratio), MACH_VALUES, rtol=1e-12
    )
    np.testing.assert_allclose(
        mach_from_pressure_ratio(pressure_ratio), MACH_VALUES, rtol=1e-12
    )
    np.testing.assert_allclose(
        mach_from_density_ratio(density_ratio), MACH_VALUES, rtol=2e-12
    )


def test_endpoint_estimators_recover_input_mach():
    upstream = np.ones(MACH_VALUES.size)
    np.testing.assert_allclose(
        mach_from_temperature_jump(
            upstream,
            [temperature_jump_from_mach(value) for value in MACH_VALUES],
        ),
        MACH_VALUES,
    )
    np.testing.assert_allclose(
        mach_from_pressure_jump(upstream, pressure_jump_from_mach(MACH_VALUES)),
        MACH_VALUES,
    )
    np.testing.assert_allclose(
        mach_from_density_jump(
            upstream,
            [density_jump_from_mach(value) for value in MACH_VALUES],
        ),
        MACH_VALUES,
        rtol=2e-12,
    )


@pytest.mark.parametrize(
    "ratio",
    [np.nan, np.inf, -np.inf, -1.0, 0.0, 0.5, 1.0],
)
def test_nonshock_or_nonfinite_ratios_are_invalid(ratio):
    assert np.isnan(mach_from_temperature_ratio(ratio))
    assert np.isnan(mach_from_pressure_ratio(ratio))
    assert np.isnan(mach_from_density_ratio(ratio))


@pytest.mark.parametrize("upstream", [0.0, -1.0, np.nan, np.inf])
def test_invalid_upstream_endpoint_is_nan(upstream):
    assert np.isnan(jump_ratio(upstream, 2.0))
    assert np.isnan(mach_from_temperature_jump(upstream, 2.0))
    assert np.isnan(mach_from_pressure_jump(upstream, 2.0))
    assert np.isnan(mach_from_density_jump(upstream, 2.0))


def test_density_estimator_does_not_clip_at_strong_shock_limit():
    strong_limit = (GAMMA + 1.0) / (GAMMA - 1.0)
    ratios = np.array(
        [strong_limit * (1.0 - 0.5e-6), strong_limit, strong_limit * 1.01]
    )
    assert np.all(np.isnan(mach_from_density_ratio(ratios)))
    assert np.isfinite(mach_from_pressure_ratio(pressure_jump_from_mach(1.0e6)))
    assert np.isfinite(
        mach_from_temperature_ratio(temperature_jump_from_mach(1.0e6))
    )


def configured_finder(**settings):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.05
    for name, value in settings.items():
        setattr(finder, name, value)
    return finder


def test_consistent_shock_populates_diagnostics_without_changing_primary_result():
    cell = planar_shock_cell(2.0, n=18, shock_index=9)
    baseline = configured_finder(validate_mach=False).find(cell)
    result = configured_finder().find(cell)
    rows = np.nonzero(result.shock)[0]

    np.testing.assert_array_equal(result.shock, baseline.shock)
    np.testing.assert_allclose(result.mach, baseline.mach)
    assert result.mach_temperature is result.mach
    assert np.all(result.pressure_check_valid[rows])
    assert np.all(result.pressure_consistent[rows])
    assert np.all(result.density_check_applicable[rows])
    assert np.all(result.density_consistent[rows])
    assert np.all(result.mach_consistent[rows])
    assert np.all(
        result.mach_validation_status[rows]
        & int(MachValidationFlag.PRESSURE_FROM_RHO_T)
    )


def test_explicit_thermal_pressure_is_preferred_and_can_fail_validation():
    cell = planar_shock_cell(2.0, n=18, shock_index=9)
    pressure = np.ones(18)
    pressure[9:] = 1.05
    cell["thermal_pressure"] = pressure

    diagnostic = configured_finder().find(cell)
    filtered = configured_finder(filter_inconsistent=True).find(cell)
    rows = np.nonzero(diagnostic.mach > 1.0)[0]

    assert rows.size
    assert np.all(diagnostic.shock[rows])
    assert np.all(diagnostic.pressure_check_valid[rows])
    assert not np.any(diagnostic.pressure_consistent[rows])
    assert not np.any(diagnostic.mach_consistent[rows])
    assert not np.any(filtered.shock[rows])
    np.testing.assert_allclose(filtered.mach[rows], diagnostic.mach[rows])
    assert not np.any(
        diagnostic.mach_validation_status[rows]
        & int(MachValidationFlag.PRESSURE_FROM_RHO_T)
    )


def test_low_mach_density_mismatch_fails_when_pressure_passes():
    cell = planar_shock_cell(2.0, n=18, shock_index=9)
    cell[("rho", "Msol/kpc3")][9:] = 1.2e6
    pressure = np.ones(18)
    pressure[9:] = pressure_jump_from_mach(2.0)
    cell["thermal_pressure"] = pressure

    result = configured_finder().find(cell)
    rows = np.nonzero(result.shock)[0]

    assert rows.size
    assert np.all(result.pressure_consistent[rows])
    assert np.all(result.density_check_applicable[rows])
    assert not np.any(result.density_consistent[rows])
    assert not np.any(result.mach_consistent[rows])


def test_high_mach_density_check_is_not_applicable():
    result = configured_finder().find(
        planar_shock_cell(5.0, n=18, shock_index=9)
    )
    rows = np.nonzero(result.shock)[0]

    assert rows.size
    assert np.all(result.pressure_consistent[rows])
    assert not np.any(result.density_check_applicable[rows])
    assert not np.any(result.density_consistent[rows])
    assert np.all(result.mach_consistent[rows])


def test_missing_endpoint_is_marked_invalid_and_untrusted():
    finder = configured_finder()
    result = shocktest.ShockResult(
        mach=np.array([2.0]),
        shock=np.array([True]),
        center_index=np.array([0]),
        upstream_index=np.array([-1]),
        downstream_index=np.array([0]),
        selected_indices=np.array([0]),
        diagnostics={},
    )
    arrays = {"temp": np.array([2.0]), "rho": np.array([2.0])}
    cell = {"thermal_pressure": np.array([2.0])}

    finder._populate_mach_validation(cell, arrays, result)

    assert not result.mach_consistent[0]
    assert result.mach_validation_status[0] & int(
        MachValidationFlag.ENDPOINT_INVALID
    )


def one_center_result(mach):
    return shocktest.ShockResult(
        mach=np.array([0.0, mach]),
        shock=np.array([False, True]),
        center_index=np.array([-1, 1]),
        upstream_index=np.array([-1, 0]),
        downstream_index=np.array([-1, 1]),
        selected_indices=np.array([0, 1]),
        diagnostics={},
    )


def test_saturated_density_is_not_applicable_and_does_not_force_failure():
    finder = configured_finder()
    result = one_center_result(5.0)
    arrays = {
        "temp": np.array([1.0, temperature_jump_from_mach(5.0)]),
        "rho": np.array([1.0, (GAMMA + 1.0) / (GAMMA - 1.0)]),
    }
    cell = {
        "thermal_pressure": np.array([1.0, pressure_jump_from_mach(5.0)])
    }

    finder._populate_mach_validation(cell, arrays, result)

    assert not result.density_check_valid[1]
    assert not result.density_check_applicable[1]
    assert result.mach_consistent[1]
    assert result.mach_validation_status[1] & int(
        MachValidationFlag.DENSITY_SATURATED
    )


def test_unavailable_pressure_never_passes_overall_validation():
    finder = configured_finder()
    result = one_center_result(2.0)
    arrays = {
        "temp": np.array([1.0, temperature_jump_from_mach(2.0)]),
        "rho": np.array([1.0, density_jump_from_mach(2.0)]),
    }
    cell = {"thermal_pressure": np.array([1.0, np.nan])}

    finder._populate_mach_validation(cell, arrays, result)

    assert not result.pressure_check_valid[1]
    assert not result.pressure_consistent[1]
    assert not result.mach_consistent[1]


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("consistency_factor", 0.9),
        ("density_check_max_mach", 1.0),
        ("density_saturation_rtol", -1.0),
    ],
)
def test_invalid_validation_settings_are_rejected(setting, value):
    finder = configured_finder(**{setting: value})
    with pytest.raises(ValueError, match=setting):
        finder.find(planar_shock_cell(2.0))
