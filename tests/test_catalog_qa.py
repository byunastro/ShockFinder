import numpy as np
import matplotlib.pyplot as plt
import shocktest
from test_maps import grid_cell


def test_quality_summary_matches_numeric_catalog():
    catalog = shocktest.ShockFinder().analyze(grid_cell()).catalog
    summary = shocktest.summarize_catalog_quality(catalog)
    assert summary['n_groups'] == len(catalog)
    assert summary['ncell'] == catalog['ncell'].sum()
    assert summary['surface_area'] == catalog['area'].sum()
    assert 'classifications' not in summary


def test_quality_plot_handles_empty_and_nonempty_catalog():
    catalog = shocktest.ShockFinder().analyze(grid_cell()).catalog
    for cat in (catalog, np.empty(0, shocktest.front_dtype)):
        figure, axes = shocktest.plot_catalog_quality(cat)
        assert axes.shape == (2, 2)
        plt.close(figure)
