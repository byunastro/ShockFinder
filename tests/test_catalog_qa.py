import matplotlib.pyplot as plt

import shocktest

from test_maps import grid_cell


def test_quality_summary_matches_catalog():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    catalog = finder.analyze(grid_cell()).catalog

    summary = shocktest.summarize_catalog_quality(catalog)

    assert summary["n_groups"] == len(catalog.groups)
    assert summary["n_complete"] + summary["n_boundary"] >= len(catalog.groups)
    assert summary["surface_area"] == sum(group.area for group in catalog.groups)
    assert sum(summary["classifications"].values()) == len(catalog.groups)


def test_quality_plot_handles_catalog_and_empty_catalog():
    finder = shocktest.ShockFinder()
    finder.minlevel = 0
    catalog = finder.analyze(grid_cell()).catalog

    figure, axes = shocktest.plot_catalog_quality(catalog)
    assert axes.shape == (2, 2)
    plt.close(figure)

    empty = shocktest.ShockCatalog(
        group_id=catalog.group_id[:0],
        center_representative=catalog.center_representative[:0],
        groups=[],
    )
    figure, axes = shocktest.plot_catalog_quality(empty)
    assert all(not axis.axison for axis in axes.ravel())
    plt.close(figure)
