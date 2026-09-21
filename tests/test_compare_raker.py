import importlib.util
from pathlib import Path

import numpy as np

from shocktest.core import ShockResult


def test_compare_raker_public_name_and_signature():
    path = Path(__file__).resolve().parents[1] / "examples" / "compare_raker.py"
    spec = importlib.util.spec_from_file_location("compare_raker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.compare_raker_result is module.compare_raker
    assert "snap_unit_kpc" in module.compare_raker.__code__.co_varnames


def test_full_snapshot_is_streamed_to_result_subset():
    path = Path(__file__).resolve().parents[1] / "examples" / "compare_raker.py"
    spec = importlib.util.spec_from_file_location("compare_raker_subset_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = np.array([0.5, 1.5, 2.5, 3.5])
    result = ShockResult(
        mach=np.array([0.0, 2.0, 3.0, 0.0]),
        shock=np.array([False, True, True, False]),
        center_index=np.full(4, -1), upstream_index=np.full(4, -1),
        downstream_index=np.full(4, -1), selected_indices=np.arange(4),
        pos=np.column_stack((x, np.full(4, 0.5), np.full(4, 0.5))) * module.KPC_IN_KM,
        dx=np.ones(4) * module.KPC_IN_KM,
        level=np.full(4, 2, dtype=np.int32),
    )
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"),
             ("Mach", "f4"), ("lv", "i4"), ("is_shock", "i4")]
    raker = np.zeros(6, dtype=dtype)
    raker["x"] = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    raker["y"] = raker["z"] = 0.5
    raker["lv"] = 2
    raker["Mach"] = [0.0, 0.0, 2.2, 0.0, 4.0, 0.0]
    raker["is_shock"] = [0, 0, 1, 0, 1, 0]

    metrics = module._subset_metrics(result, raker, 1.0, 1.0, True, 2, 1e-5)
    assert metrics["comparison_mode"] == "spatial_subset_stream"
    assert metrics["n_cells"] == 4
    assert metrics["intersection"] == 1
    assert metrics["union"] == 3
