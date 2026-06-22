from __future__ import annotations

# Based on Ryu et al. 2003, ApJ, 593, 599.

import numpy as np
import pytest

import shocktest
from shocktest import painter, pyShockFinder

from synthetic_shocks import (
    external_internal_masks,
    mach_from_temperature_jump,
    multi_shock_line_cell,
    planar_shock_cell,
    shock_surface_histogram,
    temperature_jump_from_mach,
    tiled_sheet_cell,
)


@pytest.mark.parametrize("mach", [1.5, 2.0, 3.0, 5.0, 10.0])
def test_temperature_jump_mach_recovery_for_planar_shocks(mach):
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    result = finder.find(planar_shock_cell(mach, n=18, shock_index=9))

    assert result.shock.any()
    recovered = result.mach[result.shock].max()
    assert recovered == pytest.approx(mach, rel=0.05)

    center = np.argmax(result.mach)
    assert abs(center - 9) <= 1
    assert result.upstream_index[center] < center
    assert result.downstream_index[center] > center


def test_minimum_mach_cut_rejects_weaker_temperature_jump():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    result = finder.find(planar_shock_cell(1.3, n=18, shock_index=9))

    assert not result.shock.any()
    assert np.nanmax(result.mach) == 0.0


def test_control_cases_do_not_trigger_false_shocks():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    contact = planar_shock_cell(3.0, n=18, shock_index=9)
    contact[("vx", "km/s")] = np.ones(18)
    assert not finder.find(contact).shock.any()

    rarefaction = planar_shock_cell(3.0, n=18, shock_index=9)
    rarefaction[("vx", "km/s")][:9] = -100.0
    rarefaction[("vx", "km/s")][9:] = 100.0
    assert not finder.find(rarefaction).shock.any()

    entropy_mismatch = planar_shock_cell(3.0, n=18, shock_index=9)
    entropy_mismatch[("rho", "Msol/kpc3")][9:] = 1.0e10
    assert not finder.find(entropy_mismatch).shock.any()


def test_temperature_jump_formula_matches_kernel_inverse():
    mach = np.array([1.3, 1.5, 2.0, 3.0, 5.0, 10.0])
    ratio = np.array([temperature_jump_from_mach(value) for value in mach])

    np.testing.assert_allclose(mach_from_temperature_jump(ratio), mach)


def test_external_internal_classification_and_dissipation_weighting():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    cell = multi_shock_line_cell()
    result = finder.find(cell)
    diss = pyShockFinder.compute_dissipation(cell, result)
    external, internal = external_internal_masks(cell, result)

    assert np.count_nonzero(external) >= 1
    assert np.count_nonzero(internal) >= 1
    assert result.mach[external].mean() > result.mach[internal].mean()
    assert diss.total[internal].sum() > diss.total[external].sum()


def test_surface_histogram_is_plot_ready():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    cell = tiled_sheet_cell(multi_shock_line_cell(), ny=8)
    result = finder.find(cell)
    bins = np.array([1.5, 2.0, 3.0, 5.0, 10.0])

    surface = shock_surface_histogram(result, bins=bins)

    assert surface.shape == (bins.size - 1,)
    assert np.any(surface > 0.0)
    assert np.all(np.isfinite(surface))


def test_synthetic_maps_are_plot_ready():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_mach = 1.5
    finder.show_progress = False

    cell = tiled_sheet_cell(planar_shock_cell(3.0, n=32, shock_index=16), ny=16)
    result = finder.find(cell)
    diss = pyShockFinder.compute_dissipation(cell, result)

    machmap = painter.make_mach_map(result, plane="xy", bins=(64, 32), min_mach=1.5)
    dissmap = painter.make_disspE_map(result, diss, plane="xy", bins=(64, 32), min_mach=1.5)

    assert machmap.shape == (64, 32)
    assert dissmap.shape == (64, 32)
    assert np.nanmax(machmap) == pytest.approx(3.0, rel=0.05)
    assert np.nanmax(dissmap) > 0.0


def test_thermalization_efficiency_is_monotonic_and_strong_shock_limited():
    mach = np.array([1.5, 2.0, 3.0, 5.0, 10.0, 100.0])

    efficiency = pyShockFinder.thermalization_efficiency(mach)

    assert np.all(np.diff(efficiency) > 0.0)
    assert efficiency[-1] == pytest.approx(0.56, rel=0.03)
