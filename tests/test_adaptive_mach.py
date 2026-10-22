"""Mach-dependent endpoint distances, including broadened strong shocks."""

import numpy as np
import pytest

from shocktest import ShockFinder
from shocktest.mach_validation import mach_from_temperature_jump
from synthetic_shocks import (
    cell_from_1d_profiles,
    density_jump_from_mach,
    temperature_jump_from_mach,
)


def shock_profile(mach, *, broadened=False, n=21, center=10, gamma=5.0 / 3.0):
    offset = np.arange(n, dtype=float) - center
    # A unique convergence minimum fixes the center independently of the
    # temperature transition, so endpoint distances can be checked exactly.
    velocity = -100.0 * np.clip(offset, -1.0, 1.0)
    fraction = np.clip((offset + 4.0) / 8.0, 0.0, 1.0) if broadened else (offset >= 0)
    temperature = 1.e4 * temperature_jump_from_mach(mach, gamma) ** fraction
    density = 1.e6 * density_jump_from_mach(mach, gamma) ** fraction
    return cell_from_1d_profiles(np.arange(n) + 0.5, temperature, density, velocity)


@pytest.mark.parametrize("gamma", [1.4, 5.0 / 3.0])
@pytest.mark.parametrize("mach, distance", [(3., 2), (9.999999, 2), (10., 4), (10.000001, 4), (1000., 4)])
def test_sampling_distance_switches_at_mach_ten(mach, distance, gamma):
    finder = ShockFinder()
    finder.gamma = gamma
    result = finder.find(shock_profile(mach, gamma=gamma))

    np.testing.assert_array_equal(np.flatnonzero(result.shock), [10])
    assert result.upstream_index[10] == 10 - distance
    assert result.downstream_index[10] == 10 + distance
    assert result.zone_width[10] == pytest.approx(2 * distance)
    assert result.mach[10] == pytest.approx(mach, rel=1.e-12)


@pytest.mark.parametrize("mach", [100., 1000.])
def test_four_cell_sampling_recovers_broadened_strong_shocks(mach):
    cell = shock_profile(mach, broadened=True)
    temp = cell['T', 'K']
    initial_mach = float(mach_from_temperature_jump(temp[8], temp[12]))
    assert 10. < initial_mach < mach * 0.5

    finder = ShockFinder()
    # The final Mach cut must be applied after the four-cell refinement.
    finder.min_mach = 0.9 * mach
    result = finder.find(cell)

    np.testing.assert_array_equal(np.flatnonzero(result.shock), [10])
    assert result.upstream_index[10] == 6
    assert result.downstream_index[10] == 14
    assert result.mach[10] == pytest.approx(mach, rel=1.e-12)
    assert result.temperature_ratio[10] == pytest.approx(temp[14] / temp[6])
    assert result.mach_pressure[10] == pytest.approx(mach, rel=1.e-12)


def test_weak_shock_keeps_two_cell_temperature_measurement():
    cell = shock_profile(3., broadened=True)
    result = ShockFinder().find(cell)
    temp = cell['T', 'K']
    expected = float(mach_from_temperature_jump(temp[8], temp[12]))

    np.testing.assert_array_equal(np.flatnonzero(result.shock), [10])
    assert result.upstream_index[10] == 8
    assert result.downstream_index[10] == 12
    assert result.mach[10] == pytest.approx(expected)
    assert result.mach[10] < 3.


@pytest.mark.parametrize("mach, distance", [(3., 2), (100., 4)])
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 50])
def test_safety_cap_cannot_silently_shorten_required_sampling(mach, distance, limit):
    finder = ShockFinder()
    finder.max_steps = limit
    result = finder.find(shock_profile(mach))

    if limit >= distance:
        assert result.shock[10]
        assert result.mach[10] == pytest.approx(mach)
    else:
        assert not result.shock.any()
        assert result.diagnostics['upstream_step_limit'] > 0
        assert np.all(result.upstream_index == -1)
        assert np.all(result.downstream_index == -1)


@pytest.mark.parametrize("n, center, missing_side", [(8, 3, 'upstream'), (8, 4, 'downstream')])
def test_missing_fourth_cell_rejects_strong_shock(n, center, missing_side):
    result = ShockFinder().find(shock_profile(100., n=n, center=center))

    assert not result.shock.any()
    assert result.diagnostics[f'{missing_side}_missing_neighbor'] > 0
    assert not np.any(result.mach)


def test_weak_shock_needs_no_neighbors_beyond_two_cells():
    result = ShockFinder().find(shock_profile(3., n=5, center=2))

    assert result.shock[2]
    assert result.upstream_index[2] == 0
    assert result.downstream_index[2] == 4
    assert result.mach[2] == pytest.approx(3.)


def test_temperature_floor_is_used_for_switch_and_final_measurement():
    cell = shock_profile(20.)
    finder = ShockFinder()
    finder.temperature_floor = 1.e5
    result = finder.find(cell)
    expected = float(mach_from_temperature_jump(1.e5, cell['T', 'K'][12]))

    assert expected < 10.
    assert result.shock[10]
    assert result.upstream_index[10] == 8
    assert result.downstream_index[10] == 12
    assert result.mach[10] == pytest.approx(expected)
