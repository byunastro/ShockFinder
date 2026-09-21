import numpy as np
import pytest
from shocktest import (ShockFrame, ExposureAccumulator, MergerEvent,
                       attribute_merger_shocks, integrate_galaxy_exposure)


def frame(t, x, *, flux=10., half=.5, ids=(1,), surfaces=None, normal=(1., 0., 0.)):
    n = len(ids)
    return ShockFrame(t, np.array(ids), np.array(surfaces if surfaces is not None else [7]*n),
        np.tile([x, 0., 0.], (n, 1)), np.tile(normal, (n, 1)), np.ones(n)*2,
        np.ones(n)*half, np.ones(n)*3, np.ones(n)*flux, 'analytic matched translating plane')


def integrate(previous, current, p0=(0., 0., 0.), p1=(0., 0., 0.), **kwargs):
    return integrate_galaxy_exposure([42], [p0], [p1], previous, current, **kwargs)


def test_moving_shock_crosses_stationary_galaxy_analytic_fluence():
    result = integrate(frame(0, -2), frame(4, 2))
    record, = result.records
    assert record.duration_s == pytest.approx(1.)
    assert record.fluence_erg_kpc2 == pytest.approx(10.)
    assert record.crossing_times_s == (2.,)
    assert record.attribution_status == 'unattributed'


def test_time_sampling_and_galilean_invariance():
    one = integrate(frame(0, -2), frame(4, 2, flux=30.))
    accumulator = ExposureAccumulator()
    for t in range(4):
        accumulator.add(integrate(frame(t, t-2, flux=10+5*t), frame(t+1, t-1, flux=15+5*t)))
    assert accumulator.summaries[42]['fluence_erg_kpc2'] == pytest.approx(one.records[0].fluence_erg_kpc2)
    assert accumulator.summaries[42]['crossings'] == 1
    # Add identical bulk translation to the shock and galaxy.
    boosted = integrate(frame(0, -2), frame(4, 402, flux=30.), p1=(400., 0., 0.))
    assert boosted.records == one.records
    with pytest.raises(ValueError, match='overlap'):
        accumulator.add(one)


def test_overlapping_patches_do_not_double_count_surface():
    result = integrate(frame(0, -2, ids=(1, 2)), frame(4, 2, ids=(2, 1)))
    record, = result.records
    assert record.duration_s == pytest.approx(1.)
    assert record.fluence_erg_kpc2 == pytest.approx(10.)
    assert len(record.crossing_times_s) == 1


def test_linear_flux_envelope_and_distinct_surfaces():
    previous, current = frame(0, 0, half=1., ids=(1, 2)), frame(4, 0, half=1., ids=(1, 2))
    previous.flux[:] = [0, 10]
    current.flux[:] = [10, 0]
    record, = integrate(previous, current).records
    assert record.fluence_erg_kpc2 == pytest.approx(30.)
    current.surface_id[:] = previous.surface_id[:] = [7, 8]
    result = integrate(previous, current)
    assert len(result.records) == 2
    assert sum(r.fluence_erg_kpc2 for r in result.records) == pytest.approx(40.)


def test_finite_patch_miss_zero_width_and_missing_tracking():
    assert not integrate(frame(0, -2), frame(4, 2), p0=(0., 3., 0.), p1=(0., 3., 0.)).records
    record, = integrate(frame(0, -2, half=0), frame(4, 2, half=0)).records
    assert record.duration_s == 0 and record.fluence_erg_kpc2 == 0
    assert record.crossing_times_s == (2.,)
    result = integrate(frame(0, -2), frame(4, 2, ids=(2,)))
    assert not result.records
    assert result.diagnostics['unmatched_previous'] == 1
    assert result.diagnostics['unmatched_current'] == 1


def test_normal_rotation_is_reported_not_silently_integrated():
    result = integrate(frame(0, -2), frame(4, 2, normal=(0., 1., 0.)))
    assert not result.records
    assert result.diagnostics['normal_rejected_patch_ids'] == [1]


def event(name='merger-A'):
    return MergerEvent(name, 0., 10., (0., 0., 0.), (1., 0., 0.), 1., 10., 'halo merger tree event #12')


def test_merger_attribution_evidence_ambiguity_and_exclusion():
    previous, current = frame(0, 2), frame(4, 6)
    attribution = attribute_merger_shocks(previous, current, [event()])
    assert attribution[7].status == 'candidate'
    assert attribution[7].event_ids == ('merger-A',)
    assert not attribution[7].evidence['causal_confirmation']
    ambiguous = attribute_merger_shocks(previous, current, [event(), event('merger-B')])
    assert ambiguous[7].status == 'ambiguous'
    excluded = attribute_merger_shocks(previous, current, [event()], excluded_surfaces={7: 'AGN tracer'})
    assert excluded[7].status == 'excluded'
    stationary = attribute_merger_shocks(previous, frame(4, 2), [event()])
    assert stationary[7].status == 'unattributed'
    assert attribute_merger_shocks(previous, current, [])[7].status == 'unattributed'
    interval = integrate(previous, current, p0=(4., 0., 0.), p1=(4., 0., 0.), attributions=attribution)
    accumulator = ExposureAccumulator()
    accumulator.add(interval)
    assert accumulator.summaries[42]['candidate_merger_fluence_erg_kpc2'] == pytest.approx(10.)


def test_small_workspace_and_galaxy_order_produce_same_results():
    previous, current = frame(0, -2, ids=tuple(range(40))), frame(4, 2, ids=tuple(range(40)))
    ids = np.arange(12)
    pos = np.zeros((12, 3))
    a = integrate_galaxy_exposure(ids, pos, pos, previous, current, memory_budget_bytes=4096)
    b = integrate_galaxy_exposure(ids[::-1], pos, pos, previous, current)
    assert a.records == b.records


def test_invalid_ids_and_gap_accounting():
    with pytest.raises(ValueError, match='integer'):
        integrate_galaxy_exposure([1.2], [[0, 0, 0]], [[0, 0, 0]], frame(0, -2), frame(4, 2))
    accumulator = ExposureAccumulator()
    accumulator.add(integrate(frame(0, -2), frame(4, 2)))
    accumulator.add(integrate(frame(6, 4), frame(8, 6)))
    assert accumulator.summaries[42]['gap_s'] == 2


def test_swept_sphere_broad_phase_matches_all_pairs(monkeypatch):
    import shocktest.exposure as module
    from shocktest.spatial import _blocks
    rng = np.random.default_rng(13)
    previous, current = frame(0, 0, ids=tuple(range(25))), frame(4, 0, ids=tuple(range(25)))
    previous.pos_km[:] = rng.normal(size=(25, 3))*3
    current.pos_km[:] = previous.pos_km + rng.normal(size=(25, 3))*2
    p0, p1 = rng.normal(size=(15, 3))*3, rng.normal(size=(15, 3))*3
    accelerated = integrate_galaxy_exposure(np.arange(15), p0, p1, previous, current)
    def all_pairs(p0, p1, q0, q1, radius, half_width, *, memory_budget_bytes):
        for a, b in _blocks(len(p0), len(q0), 4096):
            yield a, np.arange(b.start, b.stop)
    monkeypatch.setattr(module, 'moving_patch_blocks', all_pairs)
    reference = integrate_galaxy_exposure(np.arange(15), p0, p1, previous, current)
    assert accelerated.records == reference.records
