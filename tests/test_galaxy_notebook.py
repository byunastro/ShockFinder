"""Notebook-facing saved-front workflow with already loaded history arrays."""

import pickle

import numpy as np

from examples.galaxy import find_front_affected_galaxies
from examples.galaxy_notebook import analyze_saved_galaxies, prepare_saved_shock_search
from test_front_selection import _catalog
from test_galaxy_stripping import histories, product


def test_notebook_workflow_accepts_loaded_data_and_lazy_front_probe(tmp_path, monkeypatch):
    h, cluster = histories()
    h["r90"] = np.full(len(h["iout"]), 1.)
    with (tmp_path/"catalog_00704.pkl").open("wb") as stream:
        pickle.dump(_catalog(), stream)
    for name in ("labels", "result", "dissipation"):
        (tmp_path/f"{name}_00704.pkl").touch()

    search = prepare_saved_shock_search(
        h, cluster, tmp_path, snapshot_range=(700, 900),
        fronts=[(704, 3163)], min_front_mach=3., min_front_cells=10000,
        options={"window_gyr": .18})
    assert search.summary["mapped_snapshots"] == [704]
    assert search.front_table[0]["front_id"] == 3163
    assert search.front_table[0]["forced"]

    calls = []
    monkeypatch.setattr("examples.galaxy_notebook.open_saved_shock_snapshot",
                        lambda *args, **kwargs: (None, None, None))
    def fake_extract(*args, **kwargs):
        calls.append(kwargs["front_ids"].tolist())
        row = product(704, 200., row=3163)
        row["provenance"] = {"front_ids": [3163]}
        return row
    monkeypatch.setattr("examples.galaxy_notebook.shock_product_from_catalog", fake_extract)

    analysis, search = analyze_saved_galaxies(
        h, cluster, tmp_path, snapshots=(704,), fronts=[(704, 3163)],
        min_front_mach=3., min_front_cells=10000,
        options={"window_gyr": .18}, output_dir=tmp_path/"output")
    assert calls == [[3163]]
    assert len(analysis["classifications"]) == 1
    assert len(find_front_affected_galaxies(analysis, 704, 3163)) == 1
    assert analysis["saved_shock_selection"]["mapped_snapshots"] == [704]
    assert (tmp_path/"output"/"selected_fronts.csv").is_file()
    assert (tmp_path/"output"/"saved_shock_search.json").is_file()
