import numpy as np

import shocktest
from examples.galaxy import (
    classify_galaxy_shock_crossing,
    compact_classification_results,
    shock_front_catalog,
)
from shocktest import pyShockFinder


def test_shock_front_catalog_carries_zone_width():
    result = shocktest.ShockResult(
        mach=np.array([6.0]),
        shock=np.array([True]),
        center_index=np.array([0]),
        upstream_index=np.array([0]),
        downstream_index=np.array([0]),
        selected_indices=np.array([0]),
        pos=np.array([[0.0, 0.0, 0.0]]),
        dx=np.array([1.0]),
        normal=np.array([[1.0, 0.0, 0.0]]),
        level=np.array([0], dtype=np.int32),
        zone_width=np.array([12.0]),
    )
    dissipation = pyShockFinder.DissipationResult(
        flux=np.array([1.0]),
        total=np.array([1.0]),
        area=np.array([1.0]),
        efficiency=np.array([0.0]),
        sound_speed=np.array([0.0]),
    )

    catalog = shock_front_catalog(result, dissipation, min_mach=5.0)

    np.testing.assert_allclose(catalog["zone_width"], [12.0])


def test_classification_marks_galaxy_inside_shock_zone():
    shock_catalog = {
        "rows": np.array([10]),
        "pos": np.array([[0.0, 0.0, 0.0]]),
        "dx": np.array([1.0]),
        "mach": np.array([6.0]),
        "flux": np.array([1.0]),
        "zone_width": np.array([10.0]),
        "normal": np.array([[1.0, 0.0, 0.0]]),
        "valid_normal": np.array([True]),
    }
    galaxy_pos_prev = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    galaxy_pos_now = np.array([[4.0, 0.5, 0.0], [8.0, 0.0, 0.0]])

    classification = classify_galaxy_shock_crossing(
        galaxy_pos_prev,
        galaxy_pos_now,
        shock_catalog,
        search_radius_km=20.0,
        width_factor=2.0,
    )

    np.testing.assert_array_equal(classification["crossed"], [False, False])
    np.testing.assert_array_equal(classification["affected_zone"], [True, False])
    np.testing.assert_allclose(classification["zone_distance"], [4.0, 8.0])
    np.testing.assert_allclose(classification["zone_half_width"], [7.0, 7.0])


def test_classification_search_includes_shock_zone_width():
    shock_catalog = {
        "rows": np.array([10]),
        "pos": np.array([[0.0, 0.0, 0.0]]),
        "dx": np.array([1.0]),
        "mach": np.array([6.0]),
        "flux": np.array([1.0]),
        "zone_width": np.array([10.0]),
        "normal": np.array([[1.0, 0.0, 0.0]]),
        "valid_normal": np.array([True]),
    }

    classification = classify_galaxy_shock_crossing(
        np.array([[4.0, 0.0, 0.0]]),
        np.array([[4.0, 0.5, 0.0]]),
        shock_catalog,
        search_radius_km=1.0,
        width_factor=2.0,
    )

    np.testing.assert_array_equal(classification["near_shock"], [True])
    np.testing.assert_array_equal(classification["affected_zone"], [True])


def test_compact_classification_keeps_affected_zone_rows():
    classification = {
        "near_shock": np.array([False, False, False]),
        "crossed": np.array([False, False, False]),
        "affected_zone": np.array([False, True, False]),
    }

    compact = compact_classification_results(classification)

    np.testing.assert_array_equal(compact["galaxy_index"], [1])
    np.testing.assert_array_equal(compact["affected_zone"], [True])
