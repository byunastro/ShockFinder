"""Reduced pickle storage must preserve public arrays and scientific outputs."""
from dataclasses import fields, make_dataclass
import pickle

import numpy as np
import pytest

from shocktest import core, ShockFinder, shock_front_catalog
from shocktest.pyShockFinder import DissipationResult
from test_maps import grid_cell
from test_shock_front_catalog import saved


@pytest.mark.parametrize('protocol', [4, 5])
def test_producer_roundtrip_omits_redundancy_preserves_catalog(protocol):
    analysis = ShockFinder().analyze(grid_cell(), build_catalog=True)
    r, d = analysis.result, analysis.dissipation
    assert all(key not in r.__getstate__()[1] for key in core._VALIDATION_BITS)
    assert 'total' not in d.__getstate__()[1]
    restored_r, restored_d = pickle.loads(pickle.dumps((r, d), protocol=protocol))
    for old, new in ((r, restored_r), (d, restored_d)):
        for field in fields(old):
            value = getattr(old, field.name)
            if isinstance(value, np.ndarray):
                other = getattr(new, field.name)
                assert value.dtype == other.dtype
                assert value.shape == other.shape
                assert value.tobytes() == other.tobytes()
    assert restored_r.mach is restored_r.mach_temperature
    catalog, labels = shock_front_catalog(restored_r, restored_d, return_labels=True)
    assert catalog.tobytes() == analysis.catalog.tobytes()
    np.testing.assert_array_equal(labels, analysis.labels)


def test_all_validation_bits_and_chunk_boundaries():
    r, _ = saved()
    r.mach_validation_status = np.resize(np.arange(512, dtype=np.uint16), 131075)
    for name, bit in core._VALIDATION_BITS.items():
        setattr(r, name, (r.mach_validation_status & (1 << bit)) != 0)
    restored = pickle.loads(pickle.dumps(r))
    for name in core._VALIDATION_BITS:
        assert name not in r.__getstate__()[1]
        np.testing.assert_array_equal(getattr(restored, name), getattr(r, name))
    # Independently edited flags and explicitly disabled arrays must survive.
    r.mach_consistent[-1] ^= True
    r.pressure_consistent = None
    restored = pickle.loads(pickle.dumps(r))
    assert 'mach_consistent' in r.__getstate__()[1]
    np.testing.assert_array_equal(restored.mach_consistent, r.mach_consistent)
    assert restored.pressure_consistent is None


def test_disabled_validation_and_historical_shock_result(monkeypatch):
    r, _ = saved()
    restored = pickle.loads(pickle.dumps(r))
    assert all(getattr(restored, name) is None for name in core._VALIDATION_BITS)
    names = [f.name for f in fields(r)]
    old_class = make_dataclass('ShockResult', names, slots=True)
    old_class.__module__ = core.__name__
    r.mach_consistent = np.array([True, False, True, False])
    with monkeypatch.context() as patch:
        patch.setattr(core, 'ShockResult', old_class)
        payload = pickle.dumps(old_class(*(getattr(r, name) for name in names)))
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored.mach_consistent, r.mach_consistent)


@pytest.mark.parametrize('n', [0, 131075])
def test_total_reconstruction_and_edited_values(n):
    flux = np.resize(np.array([0., -0., np.nan, np.inf, 1.e-200, 3.]), n)
    area = np.resize(np.array([1., 1., 1., 2., 1.e-200, 7.]), n)
    with np.errstate(under='ignore'):
        total = flux * area
    d = DissipationResult(flux, total, area, np.zeros(n), np.zeros(n))
    assert 'total' not in d.__getstate__()[1]
    restored = pickle.loads(pickle.dumps(d))
    assert restored.total.tobytes() == total.tobytes()
    if n:
        d.total[-1] = 1234.
        assert 'total' in d.__getstate__()[1]
        assert pickle.loads(pickle.dumps(d)).total.tobytes() == d.total.tobytes()


def test_total_different_dtype_is_preserved():
    d = DissipationResult(np.ones(3), np.ones(3, np.float32), np.ones(3),
                          np.zeros(3), np.zeros(3))
    assert 'total' in d.__getstate__()[1]
    assert pickle.loads(pickle.dumps(d)).total.dtype == np.float32
