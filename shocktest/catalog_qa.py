"""Numeric diagnostics for generic front catalogs; no origin classification."""
import numpy as np
from .catalog_io import _validate_catalog
from . import fronts


def summarize_catalog_quality(catalog):
    catalog = _validate_catalog(catalog)
    flags = {name: int(np.count_nonzero(catalog['quality'] & getattr(fronts, name)))
             for name in ('QUALITY_NO_AREA', 'QUALITY_NO_DISS_RATE',
                          'QUALITY_UNDEFINED_NORMAL', 'QUALITY_APPROX_CONNECTIVITY',
                          'QUALITY_GAP_BRIDGED', 'QUALITY_PARTIAL_SUMMARY')}
    return dict(n_groups=len(catalog), ncell=int(catalog['ncell'].sum(dtype=np.int64)),
                surface_area=float(catalog['area'].sum()),
                dissipation_total=float(catalog['diss_rate'].sum()), quality_flags=flags)


def plot_catalog_quality(catalog, *, figsize=(12, 8)):
    import matplotlib.pyplot as plt
    catalog = _validate_catalog(catalog)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes[0, 0].hist(catalog['mach'][np.isfinite(catalog['mach'])], bins='auto')
    axes[0, 0].set(xlabel='Mean Mach', ylabel='Fronts')
    axes[0, 1].scatter(catalog['mach'], catalog['area'])
    axes[0, 1].set(xlabel='Mean Mach', ylabel='Area [kpc²]')
    axes[1, 0].scatter(catalog['ncell'], catalog['diss_rate'])
    axes[1, 0].set(xlabel='Center count', ylabel='Dissipation [erg/s]')
    flags = summarize_catalog_quality(catalog)['quality_flags']
    axes[1, 1].barh([k.removeprefix('QUALITY_') for k in flags], list(flags.values()))
    axes[1, 1].set(xlabel='Fronts')
    return fig, axes
