import numpy as np
import pytest

import shocktest


def line_cell(n=8):
    x = np.arange(n, dtype=float) + 0.5
    y = np.full(n, 0.5)
    z = np.full(n, 0.5)
    dx = np.ones(n)
    level = np.full(n, 20, dtype=np.int32)
    temp = np.full(n, 1.0e4)
    rho = np.ones(n)
    vx = np.ones(n)
    vy = np.zeros(n)
    vz = np.zeros(n)

    temp[4:] = 4.0e4
    rho[4:] = 3.0
    vx[4:] = -1.0
    return {
        ("x", "km"): x,
        ("y", "km"): y,
        ("z", "km"): z,
        ("dx", "km"): dx,
        ("vx", "km/s"): vx,
        ("vy", "km/s"): vy,
        ("vz", "km/s"): vz,
        ("T", "K"): temp,
        ("rho", "Msol/kpc3"): rho,
        "level": level,
    }


def test_planar_shock_smoke_with_tuple_keys():
    finder = shocktest.ShockFinder()
    finder.minlevel = 13
    finder.maxlevel = 20

    result = finder.ShockFinder(line_cell())

    assert result.mach.max() > 1.0
    assert result.shock.any()
    assert result.center_index[result.shock][0] >= 0
    assert result.upstream_index[result.shock][0] >= 0
    assert result.downstream_index[result.shock][0] >= 0


def test_level_filter_maps_selected_indices():
    cell = line_cell()
    cell["level"] = np.array([12, 13, 13, 14, 14, 20, 21, 22], dtype=np.int32)

    finder = shocktest.ShockFinder()
    finder.minlevel = 13
    finder.maxlevel = 20
    result = finder.find(cell)

    np.testing.assert_array_equal(result.selected_indices, np.array([1, 2, 3, 4, 5]))
    assert result.mach.shape == (5,)
    assert result.shock.shape == (5,)


def test_fine_cell_finds_coarse_face_neighbor():
    pos = np.array(
        [
            [1.0, 1.0, 1.0],  # coarse cell covers x=[0, 2]
            [2.5, 1.0, 1.0],  # fine cell covers x=[2, 3]
        ],
        dtype=float,
        order="F",
    )
    dx = np.array([2.0, 1.0])
    level = np.array([0, 1], dtype=np.int32)

    neighbors = shocktest.ShockFinder._build_neighbors(pos, dx, level)

    assert neighbors[1, 0] == 1  # fine -x neighbor is the coarse cell, 1-based for Fortran


def test_coarse_cell_records_finer_face_neighbors():
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.5, 0.5, 0.5],
            [2.5, 0.5, 1.5],
            [2.5, 1.5, 0.5],
            [2.5, 1.5, 1.5],
        ],
        dtype=float,
        order="F",
    )
    dx = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
    level = np.array([0, 1, 1, 1, 1], dtype=np.int32)

    neighbors, fine_neighbors = shocktest.ShockFinder._build_neighbor_tables(pos, dx, level)

    assert neighbors[0, 1] == 0
    np.testing.assert_array_equal(np.sort(fine_neighbors[0, 1]), np.array([2, 3, 4, 5]))


def test_missing_tuple_field_raises_clear_error():
    cell = line_cell()
    del cell[("T", "K")]
    finder = shocktest.ShockFinder()

    with pytest.raises(KeyError, match="missing required field"):
        finder.find(cell)
