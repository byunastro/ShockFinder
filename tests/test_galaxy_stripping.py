"""Analytic gas-loss and moving-front histories, without running ShockFinder."""
import pickle

import numpy as np
import pytest

from examples.galaxy import GAS_TRACERS, StrippingOptions, galaxy_stripping_catalog, save_galaxy_stripping_analysis


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


def test_default_tracer_and_moving_shock_candidate_preserve_inputs():
    h, c = histories()
    before = pickle.dumps((h, c))
    analysis = run(h, c, shocks(h))
    row = analysis["classifications"][0]
    assert row["category"] == "merger_shock_stripping_candidate"
    assert row["exposure_status"] == "shock_exposed"
    assert row["merger_shock_stripping_score"] > row["ordinary_rps_score"]
    assert row["delta_t_shock_gyr"] == pytest.approx(.15)
    assert not any(k.startswith("P_ram") for k in row)
    assert set(row["gas_loss_by_tracer"]) == {"m_gas_r90"}
    assert analysis["configuration"]["tracers"] == ("m_gas_r90",)
    events = [e for e in analysis["gas_loss_events"] if e["significant"]]
    assert len(events) == 1
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
        analysis = run(h, c, shocks(h, empty=True), tracers="m_ism")
    md = analysis["diagnostic_series"]["branch-42"]["gas_tracers"]["m_ism"]
    assert list(md["status"][[6, 7, 8]]) == ["missing", "invalid_negative", "zero_unknown_limit"]
    assert np.isnan(md["loss_rate_gyr"][[5, 6, 7, 8]]).all()
    h, c = histories()
    for tracer in GAS_TRACERS:
        h[tracer][10:] = 0.
    h["mass_limits"] = {tr: 1.e8 for tr in GAS_TRACERS}
    analysis = run(h, c, shocks(h), tracers=GAS_TRACERS)
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
    from examples.shock_catalog import (merger_shock_catalog,
                                        cache_merger_shock_inputs)
    from shocktest.core import ShockResult
    from shocktest.pyShockFinder import DissipationResult
    import shocktest
    monkeypatch.setattr(shocktest.ShockFinder, "find", lambda *a, **kw: pytest.fail("must not rerun ShockFinder"))
    h, c = histories()
    c["redshift"] = np.full(21, .67)
    c["merger_shock_options"] = {"cluster_extent_kpc": 5.,
                                 "candidate_score": .55, "thresholds_calibrated": True}
    products = {}
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
        if cached_endpoints is not None:
            result = cache_merger_shock_inputs(int(s), result, diss, tmp_path/str(s),
                                              include_endpoints=cached_endpoints, chunk_size=2)
            diss = None
        products[int(s)] = {"catalog": catalog, "result": result, "dissipation": diss, "complete": True}
    before = pickle.dumps(products)
    analysis = run(h, c, products)
    # Snapshot-local front IDs must not invent a temporally tracked crossing.
    assert not any(e["crossed"] for e in analysis["shock_encounters"])
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
    assert np.isnan(data["gas_tracers"]["m_gas_r90"]["loss_rate_gyr"][8])
    assert not data["shock_interval_covered"][8]


def test_minimal_smoothing_and_single_losing_tracer_need_no_consensus():
    h, c = histories()
    for tr in GAS_TRACERS:
        h[tr][:] = 1.e9
        h[tr][8] = 1.e8
    analysis = run(h, c, shocks(h), tracers="m_ism", smoothing_width_gyr=.15)
    np.testing.assert_allclose(analysis["diagnostic_series"]["branch-42"]["gas_tracers"]["m_ism"]["smoothed_mass"], 1.e9)
    h["m_ism"] = 1.e9*np.exp(-8.*np.clip(h["t_BB"]-7.35, 0, .15))
    for tr in GAS_TRACERS[1:]:
        h[tr][:] = 1.e9
    alone = run(h, c, shocks(h), tracers="m_ism")
    analysis = run(h, c, shocks(h), tracers=GAS_TRACERS)
    row = analysis["classifications"][0]
    assert row["category"] == "merger_shock_stripping_candidate"
    assert row["category"] == alone["classifications"][0]["category"]
    assert row["merger_shock_stripping_score"] == alone["classifications"][0]["merger_shock_stripping_score"]
    assert row["ordinary_rps_score"] == alone["classifications"][0]["ordinary_rps_score"]
    assert set(row["gas_loss_by_tracer"]) == set(GAS_TRACERS)
    assert all(row["gas_loss_by_tracer"][tr]["category"] == "no_strong_stripping" for tr in GAS_TRACERS[1:])


@pytest.mark.parametrize("selection", ["m_gas_r90", ["m_gas_r90"], ("m_gas_r90",)])
def test_explicit_selection_ignores_unselected_fields(selection):
    h, c = histories()
    baseline = run(h, c, shocks(h))
    for tracer in GAS_TRACERS:
        if tracer != "m_gas_r90":
            h[tracer] = "unselected data are not interpreted"
            h["resolved"][tracer] = "unselected mask"
    analysis = galaxy_stripping_catalog(
        {"branch-42": h}, shocks(h), c,
        options=StrippingOptions(tracers=selection, window_gyr=.18))
    assert analysis["configuration"]["tracers"] == ("m_gas_r90",)
    for key in ("classifications", "gas_loss_events", "sensitivity"):
        np.testing.assert_equal(analysis[key], baseline[key])
    assert analysis["input_histories"]["branch-42"]["m_ism"] is h["m_ism"]


def test_every_custom_tracer_is_assessed_and_uncertain_episode_is_retained():
    h, c = histories()
    h["aperture_A_mass"] = h["m_gas_r90"].copy()
    h["aperture_B_mass"] = 1.e9*np.exp(-8.*np.clip(h["t_BB"]-7.75, 0, .15))
    for tracer in GAS_TRACERS:
        del h[tracer]
    h["resolved"] = {tr: np.ones(21, bool) for tr in ("aperture_A_mass", "aperture_B_mass")}
    analysis = run(h, c, shocks(h), tracers=["aperture_A_mass", "aperture_B_mass"])
    row = analysis["classifications"][0]
    measured = row["gas_loss_by_tracer"]
    assert set(measured) == {"aperture_A_mass", "aperture_B_mass"}
    assert {e["tracer"] for e in analysis["gas_loss_events"]} == set(measured)
    assert {a["tracer"] for a in analysis["episode_assessments"]} == set(measured)
    assert all(len(m["episode_assessments"]) == 1 for m in measured.values())
    assert measured["aperture_A_mass"]["category"] == "merger_shock_stripping_candidate"
    assert measured["aperture_B_mass"]["category"] == "mixed_or_ambiguous"
    assert row["category"] == "mixed_or_ambiguous"
    assert "uncertain_gas_loss_episode" in row["ambiguity_reasons"]
    # Input order cannot hide the second tracer's episode or change the decision.
    reversed_result = run(h, c, shocks(h), tracers=["aperture_B_mass", "aperture_A_mass"])
    reversed_row = reversed_result["classifications"][0]
    for key in ("category", "confidence", "summary_gas_event_id", "merger_shock_stripping_score", "ordinary_rps_score"):
        assert reversed_row[key] == row[key]


def test_all_selected_tracers_can_reveal_different_stripping_mechanisms():
    h, c = histories()
    dt = h["t_BB"]-7.
    h["gal_cen"][:, 0] = 100.+500.*(dt-.75)**2
    h["m_ism"] = 1.e9*np.exp(-8.*np.clip(dt-.70, 0, .15))
    analysis = run(h, c, shocks(h), tracers=["m_gas_r90", "m_ism"])
    row = analysis["classifications"][0]
    assert row["gas_loss_by_tracer"]["m_gas_r90"]["category"] == "merger_shock_stripping_candidate"
    assert row["gas_loss_by_tracer"]["m_ism"]["category"] == "ordinary_infall_rps_candidate"
    assert row["category"] == "mixed_or_ambiguous"
    assert "different_episodes_favor_different_mechanisms" in row["ambiguity_reasons"]


def test_coincident_tracer_count_and_timing_agreement_do_not_weight_scores():
    h, c = histories()
    single = run(h, c, shocks(h))
    all_tracers = run(h, c, shocks(h), tracers=GAS_TRACERS)
    assert all_tracers["classifications"][0]["tracer_episode_count"] == len(GAS_TRACERS)
    assert all_tracers["classifications"][0]["episode_count"] == 1
    for key in ("category", "confidence", "merger_shock_stripping_score", "ordinary_rps_score"):
        assert all_tracers["classifications"][0][key] == single["classifications"][0][key]
    h["m_ism"] = 1.e9*np.exp(-8.*np.clip(h["t_BB"]-7.45, 0, .15))
    opts = dict(tracers=["m_gas_r90", "m_ism"], window_gyr=.3, sensitivity_fractions=(),
                sensitivity_rates_gyr=(), sensitivity_smoothing_gyr=(), sensitivity_windows_gyr=())
    separate = run(h, c, shocks(h), agreement_window_gyr=.01, **opts)
    grouped = run(h, c, shocks(h), agreement_window_gyr=.2, **opts)
    assert separate["classifications"][0]["episode_count"] == 2
    assert grouped["classifications"][0]["episode_count"] == 1
    assert grouped["classifications"][0]["tracer_temporal_agreement"] < 1.
    for key in ("category", "confidence", "merger_shock_stripping_score", "ordinary_rps_score"):
        assert separate["classifications"][0][key] == grouped["classifications"][0][key]


@pytest.mark.parametrize("orbit", [False, True])
def test_ram_pressure_never_changes_events_scores_or_decisions(orbit):
    h, c = histories(orbit=orbit, onset=.45 if orbit else .35)
    products = shocks(h, empty=orbit)
    baseline = run(h, c, products)
    for pressure in (None, np.ones(21), np.full(21, np.nan), np.full(21, -1.),
                     np.geomspace(1.e-100, 1.e100, 21), np.zeros((21, 2))):
        if pressure is None:
            h.pop("P_ram", None)
        else:
            h["P_ram"] = pressure
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            analysis = run(h, c, products)
        for key in ("gas_loss_events", "classifications", "episode_assessments", "sensitivity", "population"):
            np.testing.assert_equal(analysis[key], baseline[key])


@pytest.mark.parametrize("selection", [[], (), "", [""], ["m_gas_r90", "m_gas_r90"], ["m_gas_r90", 1], None, 1])
def test_invalid_tracer_selection_fails_explicitly(selection):
    h, c = histories()
    with pytest.raises(ValueError, match="tracers"):
        run(h, c, {}, tracers=selection)


def test_selected_columns_and_shapes_are_checked_before_loading_shocks():
    h, c = histories()
    def forbidden(iout):
        pytest.fail("invalid tracer inputs must fail before loading shock products")
    with pytest.raises(ValueError, match="branch-42.*columns missing.*custom_mass"):
        run(h, c, forbidden, tracers=["m_gas_r90", "custom_mass"])
    h["custom_mass"] = np.ones((21, 1))
    with pytest.raises(ValueError, match="custom_mass must have shape"):
        run(h, c, forbidden, tracers=["m_gas_r90", "custom_mass"])


def test_inadequate_selected_history_cannot_establish_no_stripping():
    h, c = histories()
    h["m_gas_r90"][:] = 1.e9
    h["custom_mass"] = np.full(21, np.nan)
    analysis = run(h, c, shocks(h, empty=True), tracers=["m_gas_r90", "custom_mass"])
    row = analysis["classifications"][0]
    assert row["gas_loss_by_tracer"]["m_gas_r90"]["category"] == "no_strong_stripping"
    assert row["gas_loss_by_tracer"]["custom_mass"]["category"] == "mixed_or_ambiguous"
    assert row["category"] == "mixed_or_ambiguous"
    assert "insufficient_gas_history" in row["ambiguity_reasons"]


def test_custom_tracer_plots_with_optional_pressure_and_sorted_history():
    import matplotlib.pyplot as plt
    from examples.galaxy import plot_galaxy_stripping
    h, c = histories()
    h["custom_mass"] = h["m_gas_r90"].copy()
    h["resolved"]["custom_mass"] = np.ones(21, bool)
    h.pop("P_ram")
    for key, value in list(h.items()):
        if isinstance(value, np.ndarray):
            h[key] = value[::-1].copy()
    for key in ("resolved",):
        h[key] = {tr: value[::-1].copy() for tr, value in h[key].items()}
    products = {int(s): product(int(s), 100.+500.*(t-7.), row=j) for j, (s, t) in enumerate(zip(h["iout"], h["t_BB"]))}
    analysis = run(h, c, products, tracers=["custom_mass", "m_gas_r90"])
    figure, axes = plot_galaxy_stripping(analysis, "branch-42")
    try:
        assert {line.get_label() for line in axes[1].lines if not line.get_label().startswith("_")} == {"custom_mass", "m_gas_r90"}
        assert np.isnan(axes[3].lines[0].get_ydata()).all()
        assert analysis["classifications"][0]["category"] == "merger_shock_stripping_candidate"
        figure.canvas.draw()
    finally:
        plt.close(figure)
