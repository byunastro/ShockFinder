import numpy as np
import pytest

import shocktest
from shocktest.core import _shockfinder


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


def test_temperature_and_density_filters_map_selected_indices():
    cell = line_cell()
    cell[("T", "K")] = np.array([1e4, 2e5, 2e5, 2e5, 4e5, 4e5, 4e5, 4e5], dtype=float)
    cell[("rho", "Msol/kpc3")] = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0], dtype=float)

    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    finder.maxlevel = 20
    finder.min_temperature = 2e5
    finder.min_density = 1.0
    finder.max_density = 5.0
    result = finder.find(cell)

    np.testing.assert_array_equal(result.selected_indices, np.array([2, 3, 4, 5, 6]))
    assert result.mach.shape == (5,)


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


def test_multi_level_fine_face_gap_raises_clear_error():
    pos = np.array(
        [
            [2.0, 2.0, 2.0],
            [4.5, 0.5, 0.5],
        ],
        dtype=float,
        order="F",
    )
    dx = np.array([4.0, 1.0])
    level = np.array([0, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="one refinement jump"):
        shocktest.ShockFinder._build_neighbor_tables(pos, dx, level)


def test_shock_center_considers_finer_face_candidates():
    pos = np.asfortranarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    vel = np.asfortranarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [-20.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -100.0, 0.0],
        ],
        dtype=float,
    )
    dx = np.asfortranarray(np.ones(8, dtype=float))
    temp = np.asfortranarray(
        [1.0e4, 1.0e4, 2.0e4, 4.0e4, 8.0e4, 16.0e4, 4.0e4, 8.0e4], dtype=float
    )
    rho = np.asfortranarray([1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 2.0, 3.0], dtype=float)
    level = np.asfortranarray([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int32)
    neighbors = np.zeros((8, 6), dtype=np.int32, order="F")
    fine_neighbors = np.zeros((8, 6, 4), dtype=np.int32, order="F")

    # Cell 3 is a coarse candidate with a finer +x face candidate at cell 4.
    # Cells 4 and 5 are progressively more convergent. Center selection must
    # continue through the finer neighbor and settle on cell 5, not stop at 4.
    neighbors[2, 0] = 2
    fine_neighbors[2, 1, 0] = 4
    neighbors[3, 0] = 3
    neighbors[3, 1] = 5
    neighbors[4, 0] = 4
    neighbors[4, 1] = 6
    # Cell 7 is a much more convergent candidate in the tangential +y
    # direction. It must not pull the center away from the x-directed normal.
    neighbors[2, 3] = 7
    neighbors[6, 2] = 3
    neighbors[6, 3] = 8

    mach, shock, center, upstream, downstream, diagnostics = (
        _shockfinder.shockfinder_kernel.find_shocks(
        pos,
        vel,
        dx,
        temp,
        rho,
        level,
        neighbors,
        fine_neighbors,
        5.0 / 3.0,
        1.0e4,
        1.0,
        3,
        50,
        0.7,
        1.0e-12,
        0,
        0,
        8,
        )
    )

    assert shock[2] == 0
    assert shock[3] == 0
    assert shock[4] == 1
    assert center[4] == 5
    assert upstream[4] == 2
    assert downstream[4] == 6
    assert mach[4] > 1.0
    assert diagnostics.shape == (10,)

    plateau_vel = vel.copy(order="F")
    plateau_vel[5, 0] = -10.5  # Cells 4 and 5 now have identical divV=-5.
    _, plateau_shock, plateau_center, _, _, _ = (
        _shockfinder.shockfinder_kernel.find_shocks(
            pos,
            plateau_vel,
            dx,
            temp,
            rho,
            level,
            neighbors,
            fine_neighbors,
            5.0 / 3.0,
            1.0e4,
            1.0,
            3,
            50,
            0.7,
            1.0e-12,
            0,
            0,
            8,
        )
    )
    np.testing.assert_array_equal(np.nonzero(plateau_shock)[0], [3])
    assert plateau_center[3] == 4

    _, _, _, _, _, center_capped_diagnostics = (
        _shockfinder.shockfinder_kernel.find_shocks(
            pos,
            vel,
            dx,
            temp,
            rho,
            level,
            neighbors,
            fine_neighbors,
            5.0 / 3.0,
            1.0e4,
            1.0,
            3,
            1,
            0.7,
            1.0e-12,
            0,
            0,
            8,
        )
    )
    assert center_capped_diagnostics[0] > 0

    # With only two steps, the upstream walk is still inside the candidate
    # zone. Reaching the cap must reject the detection instead of using that
    # candidate cell as a pre-shock endpoint.
    _, capped_shock, _, _, _, capped_diagnostics = (
        _shockfinder.shockfinder_kernel.find_shocks(
        pos,
        vel,
        dx,
        temp,
        rho,
        level,
        neighbors,
        fine_neighbors,
        5.0 / 3.0,
        1.0e4,
        1.0,
        2,
        50,
        0.7,
        1.0e-12,
        0,
        0,
        8,
        )
    )
    assert not np.any(capped_shock)
    assert capped_diagnostics[2] > 0


def test_default_shock_walk_limit_is_fifty_steps():
    assert shocktest.ShockFinder().max_steps == 50


def test_missing_tuple_field_raises_clear_error():
    cell = line_cell()
    del cell[("T", "K")]
    finder = shocktest.ShockFinder()

    with pytest.raises(KeyError, match="missing required field"):
        finder.find(cell)


def test_only_open_boundary_is_supported_for_extracted_regions():
    finder = shocktest.ShockFinder()
    assert finder.boundary == "open"

    finder.boundary = "periodic"
    with pytest.raises(ValueError, match="boundary must be 'open'"):
        finder.find(line_cell())


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("gamma", 1.0, "gamma"),
        ("gamma", np.nan, "gamma"),
        ("temperature_floor", 0.0, "temperature_floor"),
        ("min_mach", 0.9, "min_mach"),
        ("max_steps", -1, "max_steps"),
        ("max_steps", 1.5, "max_steps"),
        ("max_center_steps", 0, "max_center_steps"),
        ("max_center_steps", 1.5, "max_center_steps"),
        ("center_normal_cosine", -0.1, "center_normal_cosine"),
        ("center_normal_cosine", 1.1, "center_normal_cosine"),
        ("center_plateau_tolerance", -1.0, "center_plateau_tolerance"),
    ],
)
def test_invalid_physical_settings_are_rejected(attribute, value, message):
    finder = shocktest.ShockFinder()
    setattr(finder, attribute, value)

    with pytest.raises(ValueError, match=message):
        finder.find(line_cell())


def test_large_walk_limit_warns_but_is_used():
    finder = shocktest.ShockFinder()
    finder.max_steps = 51

    with pytest.warns(RuntimeWarning, match="exceeds.*50"):
        result = finder.find(line_cell())

    assert result.shock.any()
    assert result.diagnostics is not None
    assert set(result.diagnostics) == {
        "center_step_limit",
        "upstream_missing_neighbor",
        "upstream_step_limit",
        "downstream_missing_neighbor",
        "downstream_step_limit",
        "candidate_exit",
        "thermodynamic_exit",
        "nonconverging_exit",
        "invalid_jump",
        "mach_rejected",
    }


@pytest.mark.parametrize(
    "field",
    [
        ("x", "km"),
        ("dx", "km"),
        ("vx", "km/s"),
        ("T", "K"),
        ("rho", "Msol/kpc3"),
        "level",
    ],
)
def test_nonfinite_input_fields_are_rejected(field):
    cell = line_cell()
    cell[field] = np.asarray(cell[field], dtype=float)
    cell[field][2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        shocktest.ShockFinder().find(cell)


def test_shock_result_clear_releases_arrays():
    finder = shocktest.ShockFinder()
    result = finder.ShockFinder(line_cell())

    result.clear()
    finder.clear()

    assert result.mach.size == 0
    assert result.shock.size == 0
    assert result.selected_indices.size == 0
    assert result.pos is None
    assert result.dx is None
    assert result.normal is None
    assert result.level is None
    assert result.zone_width is None
