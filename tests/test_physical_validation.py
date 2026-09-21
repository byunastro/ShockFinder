"""Scale, ordering, orientation and AMR regressions for the physical detector."""
import numpy as np
import pytest

from shocktest import ShockFinder
from shocktest.pyShockFinder import compute_dissipation
from synthetic_shocks import temperature_jump_from_mach, density_jump_from_mach

KPC_KM = 3.0856775814913673e16


def shock_mesh(pos, dx, level, normal=(1., 0., 0.), mach=3., smooth=False):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    distance = pos @ normal
    f = (1 + np.tanh(distance / 1.5)) / 2 if smooth else (distance >= 0).astype(float)
    rho = 1 + f * (density_jump_from_mach(mach) - 1)
    temperature = 1.e7 * (1 + f * (temperature_jump_from_mach(mach) - 1))
    velocity = 1000 * normal[None, :] / rho[:, None]
    return {
        **{(axis, 'km'): pos[:, i] for i, axis in enumerate('xyz')},
        **{('v' + axis, 'km/s'): velocity[:, i] for i, axis in enumerate('xyz')},
        ('dx', 'km'): np.asarray(dx), 'level': np.asarray(level, dtype=np.int32),
        ('T', 'K'): temperature, ('rho', 'Msol/kpc3'): rho,
    }


def regular_mesh(normal=(1., 0., 0.), smooth=False):
    pos = np.stack(np.meshgrid(*([np.arange(-10, 10) + .5] * 3), indexing='ij'), axis=-1).reshape(-1, 3)
    return shock_mesh(pos, np.ones(len(pos)), np.full(len(pos), 19), normal, smooth=smooth)


def signature(result):
    rows = np.flatnonzero(result.shock)
    order = np.lexsort(result.pos[rows].T[::-1])
    rows = rows[order]
    return result.pos[rows], result.mach[rows]


@pytest.mark.parametrize('smooth', [False, True])
def test_physical_scale_and_row_permutation(smooth):
    cell = regular_mesh(smooth=smooth)
    finder = ShockFinder()
    baseline = signature(finder.find(cell))
    assert len(baseline[0]) > 0
    permutation = np.random.default_rng(92).permutation(len(cell['level']))
    for scale in (1., KPC_KM):
        other = {k: v[permutation].copy() for k, v in cell.items()}
        for field in ('x', 'y', 'z', 'dx'):
            other[field, 'km'] *= scale
        result = finder.find(other)
        positions, mach = signature(result)
        np.testing.assert_allclose(positions / scale, baseline[0], atol=1.e-12)
        np.testing.assert_allclose(mach, baseline[1], rtol=1.e-10)


@pytest.mark.parametrize('normal', [(1., 0., 0.), (1., .43, .23), (1., 1., 1.)])
def test_oblique_shock_recovers_mach(normal):
    result = ShockFinder().find(regular_mesh(normal))
    assert np.count_nonzero(result.shock) > 20
    np.testing.assert_allclose(result.mach[result.shock], 3., rtol=.01)
    normals = result.normal[result.shock]
    direction = np.asarray(normal) / np.linalg.norm(normal)
    assert np.median(normals @ direction) > .85


def test_refinement_interface_mach_and_ordering():
    coarse = np.stack(np.meshgrid(np.arange(-7, 0, 2), np.arange(-7, 8, 2), np.arange(-7, 8, 2), indexing='ij'), axis=-1).reshape(-1, 3)
    fine = np.stack(np.meshgrid(np.arange(.5, 8), np.arange(-7.5, 8), np.arange(-7.5, 8), indexing='ij'), axis=-1).reshape(-1, 3)
    pos = np.concatenate((coarse, fine))
    cell = shock_mesh(pos, np.r_[np.full(len(coarse), 2.), np.ones(len(fine))], np.r_[np.full(len(coarse), 18), np.full(len(fine), 19)])
    finder = ShockFinder()
    result = finder.find(cell)
    assert result.shock.any()
    np.testing.assert_allclose(result.mach[result.shock], 3., rtol=.01)
    order = np.random.default_rng(12).permutation(len(pos))
    shuffled = finder.find({k: v[order] for k, v in cell.items()})
    for left, right in zip(signature(result), signature(shuffled)):
        np.testing.assert_allclose(left, right)


def test_selection_does_not_truncate_endpoint_measurement():
    cell = regular_mesh(smooth=True)
    finder = ShockFinder()
    full = finder.find(cell)
    finder.min_temperature = 2.e7
    selected = finder.find(cell)
    assert selected.shock.any()
    rows = selected.shock
    np.testing.assert_allclose(selected.mach[rows], full.mach[rows])
    np.testing.assert_array_equal(selected.upstream_index[rows], full.upstream_index[rows])


def test_measurement_settings_follow_result():
    cell = regular_mesh()
    finder = ShockFinder()
    finder.temperature_floor = 1.2e7
    analysis = finder.analyze(cell, build_catalog=False)
    standalone = compute_dissipation(cell, analysis.result)
    np.testing.assert_allclose(standalone.flux, analysis.dissipation.flux)
    assert analysis.result.temperature_floor == finder.temperature_floor
    with pytest.raises(ValueError, match='match'):
        compute_dissipation(cell, analysis.result, temperature_floor=1.e4)


def test_coarse_face_gradient_uses_fine_center_position():
    from shocktest.core import _shockfinder
    coarse = np.stack(np.meshgrid(np.arange(-7, 0, 2), np.arange(-7, 8, 2), np.arange(-7, 8, 2), indexing='ij'), axis=-1).reshape(-1, 3)
    fine = np.stack(np.meshgrid(np.arange(.5, 8), np.arange(-7.5, 8), np.arange(-7.5, 8), indexing='ij'), axis=-1).reshape(-1, 3)
    pos = np.concatenate((coarse, fine))
    dx = np.r_[np.full(len(coarse), 2.), np.ones(len(fine))]
    level = np.r_[np.full(len(coarse), 18), np.full(len(fine), 19)]
    cell = shock_mesh(pos, dx, level)
    cell['T', 'K'] = 1.e7 + 100. * (pos @ [1., 2., 3.])
    finder = ShockFinder()
    arrays = finder._extract_amr_arrays(cell)
    neighbors, fine_index, fine = finder._neighbor_tables_for_arrays(arrays['pos'], arrays['dx'], arrays['level'])
    row = np.flatnonzero(np.all(pos == [-1, 1, 1], axis=1))[0]
    normals = _shockfinder.shockfinder_kernel.shock_normals(arrays['pos'], arrays['vel'], arrays['dx'],
        arrays['temp'], arrays['rho'], neighbors, fine_index, fine, np.array([row+1], dtype=np.int32),
        finder.gamma, len(pos), len(fine), 1)
    np.testing.assert_allclose(normals[0], np.array([1., 2., 3.]) / np.sqrt(14), rtol=1e-10)
