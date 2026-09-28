"""Analytic gas-loss and moving-front histories, without running ShockFinder."""
import pickle

import numpy as np
import pytest

from examples.galaxy import GAS_TRACERS, galaxy_stripping_catalog, save_galaxy_stripping_analysis


def histories(*, orbit=False, onset=.35, invalid=False):
    dt = np.arange(21)*.05
    t = 7.+dt
    snap = np.arange(700, 721)
    mass = 1.e9*np.exp(-8.*np.clip(dt-onset, 0, .15))
    position = np.column_stack((100.+500.*(dt-.5)**2 if orbit else np.full(21, 200.), np.zeros(21), np.zeros(21)))
    h = {"iout": snap, "t_BB": t, "gal_cen": position, "gas_radius": np.full(21, 1.),
         "P_ram": np.exp(2.*dt) if orbit else np.where(dt < onset, 1., 5.),
         "vx": np.zeros(21), "vy": np.zeros(21), "vz": np.zeros(21),
         "hmid": np.arange(1000, 1021), "first": np.full(21, 1000), "last": np.full(21, 2000),
         "host_cluster": 1, "ism_flag": np.full(21, 37), "icm_flag": np.full(21, 52),
         "sfr_raw": np.linspace(3., 0., 21), "units": {"position": "kpc", "radius": "kpc", "time": "Gyr",
         "velocity": "km/s", "coordinate_frame": "physical"},
         "resolved": {tr: np.ones(21, bool) for tr in GAS_TRACERS}}
    for tr in GAS_TRACERS:
        h[tr] = mass.copy()
    if invalid:
        h["m_ism"][[6, 7, 8]] = [np.nan, -1., 0.]
    c = {"iout": snap, "t_BB": t, "ccen1": np.zeros((21, 3)),
         "ccen2": np.column_stack((300.+1000.*(dt-.25)**2, np.zeros(21), np.zeros(21))),
         "rvir1": np.full(21, 1000.), "rvir2": np.full(21, 800.)}
    return h, c


def product(snapshot, x, *, row=10, track="700:10", classification="candidate", complete=True, dx=.5):
    pos = np.array([[x, 0., 0.]])
    member = {"iout": snapshot, "shock_id": np.array([row]), "position_unit": "kpc",
              "pos": pos, "dx": np.array([dx]), "normal": np.array([[1., 0., 0.]]),
              "mach": np.array([3.]), "flux": np.array([1.e39]), "total": np.array([1.e39*dx**2]),
              "area": np.array([dx**2]), "zone_width": np.array([0.]),
              "upstream_pos": pos-np.array([[dx, 0., 0.]]),
              "downstream_pos": pos+np.array([[dx, 0., 0.]]),
              "dissipation_units": {"flux": "erg/s/kpc2", "total": "erg/s", "area": "kpc2"},
              "assessment": {"shock_id": row, "track_origin": track, "classification": classification,
                             "evidence": .90, "confidence": "medium", "quality_flags": ["uncalibrated_evidence_score"]}}
    return {"members": [member], "complete": complete}


def shocks(h, *, empty=False):
    return {int(s): {"members": [], "complete": True} if empty else product(int(s), 100.+500.*(t-7.), row=10+j)
            for j, (s, t) in enumerate(zip(h["iout"], h["t_BB"]))}


def run(h, c, products, **options):
    return galaxy_stripping_catalog({"branch-42": h}, products, c, options={"window_gyr": .18, **options})


def test_independent_tracers_and_moving_shock_candidate_preserve_inputs():
    h, c = histories()
    before = pickle.dumps((h, c))
    analysis = run(h, c, shocks(h))
    row = analysis["classifications"][0]
    assert row["category"] == "merger_shock_stripping_candidate"
    assert row["exposure_status"] == "shock_exposed"
    assert row["merger_shock_stripping_score"] > row["ordinary_rps_score"]
    assert row["delta_t_shock_gyr"] == pytest.approx(.15)
    assert row["P_ram_enhancement"] == pytest.approx(5.)
    assert set(row["gas_loss_by_tracer"]) == set(GAS_TRACERS)
    events = [e for e in analysis["gas_loss_events"] if e["significant"]]
    assert len(events) == 6
    assert all(e["fractional_loss"] == pytest.approx(1.-np.exp(-1.2)) for e in events)
    assert all(e["timescale_gyr"] == pytest.approx(1./8.) for e in events)
    enc = next(e for e in analysis["shock_encounters"] if e["crossed"])
    assert enc["peak_time_gyr"] == pytest.approx(7.2)
    assert len({s[1] for s in enc["shock_ids"]}) > 1  # AMR indices are not tracked IDs.
    assert not row["causal_confirmation"]
    assert pickle.dumps((h, c)) == before
    assert analysis["input_histories"]["branch-42"]["sfr_raw"] is h["sfr_raw"]


def test_ordinary_rps_needs_covered_absence_not_missing_shock_products():
    h, c = histories(orbit=True, onset=.45)
    covered = run(h, c, shocks(h, empty=True))
    row = covered["classifications"][0]
    assert row["category"] == "ordinary_infall_rps_candidate"
    assert row["pericenter_time_gyr"] == pytest.approx(7.5)
    assert row["delta_t_peri_gyr"] == pytest.approx(-.05)
    assert row["P_ram_smooth_evidence"] > row["P_ram_sudden_evidence"]
    unknown = run(h, c, {})["classifications"][0]
    assert unknown["category"] == "mixed_or_ambiguous"
    assert unknown["exposure_status"] == "unknown_or_uncertain_exposure"
    assert "shock_coverage_incomplete" in unknown["quality_flags"]


def test_pericenter_and_crossing_coincidence_remains_ambiguous():
    h, c = histories()
    dt = h["t_BB"]-7.
    h["gal_cen"][:, 0] = 200.+200.*(dt-.2)**2
    analysis = run(h, c, shocks(h))
    row = analysis["classifications"][0]
    assert row["category"] == "mixed_or_ambiguous"
    assert "shock_and_pericenter_times_unresolved" in row["ambiguity_reasons"]


def test_one_snapshot_proximity_and_untracked_ids_cannot_confirm_crossing():
    h, c = histories()
    lone = run(h, c, {704: product(704, 200.)})
    assert len(lone["shock_encounters"]) == 1
    assert not lone["shock_encounters"][0]["confirmed"]
    assert lone["classifications"][0]["category"] == "mixed_or_ambiguous"
    changing = {int(s): product(int(s), 100.+500.*(t-7.), track=f"{s}:10") for s, t in zip(h["iout"], h["t_BB"])}
    analysis = run(h, c, changing)
    assert not any(e["confirmed"] for e in analysis["shock_encounters"])


def test_invalid_values_never_logged_and_censored_losses_are_bounds():
    h, c = histories(invalid=True)
    with np.errstate(divide="raise", invalid="raise"):
        analysis = run(h, c, shocks(h, empty=True))
    md = analysis["diagnostic_series"]["branch-42"]["gas_tracers"]["m_ism"]
    assert list(md["status"][[6, 7, 8]]) == ["missing", "invalid_negative", "zero_unknown_limit"]
    assert np.isnan(md["loss_rate_gyr"][[5, 6, 7, 8]]).all()
    h, c = histories()
    for tracer in GAS_TRACERS:
        h[tracer][10:] = 0.
    h["mass_limits"] = {tr: 1.e8 for tr in GAS_TRACERS}
    analysis = run(h, c, shocks(h))
    bounds = [e for e in analysis["gas_loss_events"] if e["fraction_is_lower_bound"]]
    assert len(bounds) == 6
    assert all(e["timescale_is_upper_bound"] for e in bounds)
    assert all(e["fractional_loss"] >= .89 for e in bounds)


def test_periodic_amr_radius_and_between_output_crossing():
    h, c = histories()
    h["gal_cen"][:, 0] = 99.
    c["box_size"] = 100.
    c["ccen2"][:, 0] = 40.
    h["gas_radius"][:] = .1
    products = {int(s): product(int(s), (94.+100.*(t-7.)) % 100., dx=.2, row=20+j) for j, (s, t) in enumerate(zip(h["iout"], h["t_BB"]))}
    analysis = run(h, c, products)
    assert any(e["crossed"] and e["peak_time_gyr"] == pytest.approx(7.05) for e in analysis["shock_encounters"])
    # A shock crosses a stationary galaxy between outputs with no sampled overlap.
    products = {700: product(700, 98., dx=.2), 701: product(701, 2., row=111, dx=.2)}
    analysis = run(h, c, products)
    enc = next(e for e in analysis["shock_encounters"] if e["crossed"])
    assert enc["peak_time_gyr"] == pytest.approx(7.0125)
    assert np.isfinite(enc["relative_motion_normal_angle_deg"])


def test_secondary_branch_ends_without_fabricated_pericenter():
    h, c = histories()
    h["host_cluster"] = 2
    c["ccen2"][11:] = np.nan
    analysis = run(h, c, shocks(h, empty=True))
    data = analysis["diagnostic_series"]["branch-42"]
    assert np.isnan(data["secondary_distance_kpc"][11:]).all()
    assert not any(p["snapshot"] >= 710 and p["verified_turning_point"] for p in data["pericenters"])


def test_threshold_sensitivity_and_flat_or_transient_histories():
    h, c = histories()
    analysis = run(h, c, shocks(h), sensitivity_fractions=(.2, .3, .9))
    high = next(r for r in analysis["sensitivity"] if r["variant"] == "min_fractional_loss=0.9")
    assert high["category"] == "no_strong_stripping"
    assert high["episode_count"] == 0
    assert analysis["classifications"][0]["classification_stability"] < 1.
    for tr in GAS_TRACERS:
        h[tr][:] = 1.e9
    flat = run(h, c, shocks(h))
    assert flat["classifications"][0]["category"] == "no_strong_stripping"
    for tr in GAS_TRACERS:
        h[tr][8] = 1.e8
    transient = run(h, c, shocks(h))
    assert transient["classifications"][0]["category"] == "no_strong_stripping"
    assert any("transient_rebound" in e["quality_flags"] for e in transient["gas_loss_events"])


def test_units_required_and_streaming_outputs_round_trip(tmp_path):
    h, c = histories()
    h["units"].pop("coordinate_frame")
    with pytest.raises(ValueError, match="declare physical"):
        run(h, c, {})
    h["units"]["coordinate_frame"] = "physical"
    calls = []
    products = shocks(h)
    def load(iout):
        calls.append(iout)
        return products[iout]
    analysis = run(h, c, load)
    assert calls == list(h["iout"])
    paths = save_galaxy_stripping_analysis(analysis, tmp_path)
    with open(paths["analysis"], "rb") as stream:
        restored = pickle.load(stream)
    assert restored["classifications"][0]["category"] == analysis["classifications"][0]["category"]
    assert restored["classifications"][0]["delta_t_shock_gyr"] == pytest.approx(.15)
    np.testing.assert_array_equal(restored["input_histories"]["branch-42"]["sfr_raw"], h["sfr_raw"])
    assert (tmp_path/"diagnostics"/"population.png").exists()
    assert not (tmp_path/"README.md").exists()


@pytest.mark.parametrize("cached_endpoints", [None, True, False])
def test_adapter_reads_native_merger_catalog_and_saved_km_cells(monkeypatch, tmp_path, cached_endpoints):
    from examples.shock_catalog import (merger_shock_catalog, finalize_merger_shock_catalogs,
                                        cache_merger_shock_inputs)
    from shocktest.core import ShockResult
    from shocktest.pyShockFinder import DissipationResult
    import shocktest
    monkeypatch.setattr(shocktest.ShockFinder, "find", lambda *a, **kw: pytest.fail("must not rerun ShockFinder"))
    h, c = histories()
    c["redshift"] = np.full(21, .67)
    c["merger_shock_options"] = {"cluster_extent_kpc": 5., "minimum_track_length": 2,
                                 "candidate_score": .55, "thresholds_calibrated": True}
    products, catalogs = {}, []
    for s, t in zip(h["iout"], h["t_BB"]):
        x = 100.+500.*(t-7.)
        pos = np.array([[x-4., 0, 0], [x, -4., 0], [x, 0., 0], [x, 4., 0], [x+4., 0, 0]])*3.0856775814913673e16
        mask = np.array([False, True, True, True, False])
        dense = np.array([-1, 1, 2, 3, -1], np.int32)
        result = ShockResult(np.where(mask, 3., 0.), mask, dense, np.where(mask, 0, -1).astype(np.int32),
                             np.where(mask, 4, -1).astype(np.int32), np.arange(5)+1000,
                             pos=pos, dx=np.full(5, 4.*3.0856775814913673e16),
                             normal=np.tile([1., 0, 0], (5, 1)), mach_consistent=mask, position_unit="km")
        diss = DissipationResult(np.full(5, 1.e39), np.full(5, 16.e39), np.full(5, 16.), np.zeros(5), np.zeros(5))
        catalog = merger_shock_catalog(int(s), result, diss, c)
        catalogs.append(catalog)
        c["previous_catalog"] = catalog
        if cached_endpoints is not None:
            result = cache_merger_shock_inputs(int(s), result, diss, tmp_path/str(s),
                                              include_endpoints=cached_endpoints, chunk_size=2)
            diss = None
        products[int(s)] = {"catalog": catalog, "result": result, "dissipation": diss, "complete": True}
    for catalog in finalize_merger_shock_catalogs(catalogs):
        products[catalog["iout"]]["catalog"] = catalog
    before = pickle.dumps(products)
    analysis = run(h, c, products)
    assert any(e["crossed"] for e in analysis["shock_encounters"])
    data = analysis["diagnostic_series"]["branch-42"]
    assert data["shock_distance_kpc"][4] == pytest.approx(0., abs=1.e-10)
    assert data["shock_cell_size_kpc"][4] == pytest.approx(4.)
    assert data["shock_id"][4] in (1, 2, 3)
    assert data["shock_side"][4] == ("unknown" if cached_endpoints is False else "straddling")
    assert pickle.dumps(products) == before


def test_no_derivatives_or_crossings_across_important_history_gap():
    h, c = histories()
    h["t_BB"][9:] += .5
    c["t_BB"] = h["t_BB"].copy()
    for tracer in GAS_TRACERS:
        h[tracer][:9], h[tracer][9:] = 1.e9, 1.e8
    analysis = run(h, c, shocks(h, empty=True))
    assert not any(e["significant"] for e in analysis["gas_loss_events"])
    assert analysis["classifications"][0]["category"] == "mixed_or_ambiguous"
    data = analysis["diagnostic_series"]["branch-42"]
    assert np.isnan(data["gas_tracers"]["m_ism"]["loss_rate_gyr"][8])
    assert not data["shock_interval_covered"][8]


def test_minimal_smoothing_removes_snapshot_dip_and_tracer_disagreement_flags():
    h, c = histories()
    for tr in GAS_TRACERS:
        h[tr][:] = 1.e9
        h[tr][8] = 1.e8
    analysis = run(h, c, shocks(h), smoothing_width_gyr=.15)
    np.testing.assert_allclose(analysis["diagnostic_series"]["branch-42"]["gas_tracers"]["m_ism"]["smoothed_mass"], 1.e9)
    h["m_ism"] = 1.e9*np.exp(-8.*np.clip(h["t_BB"]-7.35, 0, .15))
    for tr in GAS_TRACERS[1:]:
        h[tr][:] = 1.e9
    analysis = run(h, c, shocks(h))
    assert analysis["classifications"][0]["category"] == "mixed_or_ambiguous"
    assert "gas_tracers_disagree_or_insufficient_independent_families" in analysis["classifications"][0]["ambiguity_reasons"]
