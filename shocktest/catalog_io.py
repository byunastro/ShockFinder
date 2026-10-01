"""Numeric front catalogs and optional original-record membership, without pickle."""
import csv
from pathlib import Path
import numpy as np
from .fronts import front_dtype

CATALOG_SCHEMA_VERSION = 3


def _validate_catalog(catalog, labels=None):
    catalog = np.asarray(catalog)
    if catalog.ndim != 1 or catalog.dtype != front_dtype:
        raise ValueError('catalog must be a one-dimensional front_dtype array')
    if not np.array_equal(catalog['front_id'], np.arange(len(catalog))):
        raise ValueError('front IDs must be sequential from zero')
    if np.any(catalog['ncell'] < 1):
        raise ValueError('front ncell must be positive')
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.dtype != np.dtype('int32'):
            raise ValueError('labels must be a one-dimensional int32 array')
        counts = np.zeros(len(catalog), np.int64)
        for start in range(0, len(labels), 131072):
            block = labels[start:start+131072]
            if np.any(block < -1) or np.any(block >= len(catalog)):
                raise ValueError('labels contain invalid front IDs')
            # Avoid full-result boolean/index temporaries during archive checks.
            active, number = np.unique(block[block >= 0], return_counts=True)
            counts[active] += number
        if not np.array_equal(counts, catalog['ncell']):
            raise ValueError('membership counts do not match catalog ncell')
    return catalog


def save_shock_catalog(path, catalog, *, labels=None, compressed=True):
    """Save 78-byte rows, plus optional int32 membership in result-record order.

    Keep this archive paired with its original result; row order is not inferred
    on reload. Membership cannot be reconstructed from summary rows alone.
    """
    catalog = _validate_catalog(catalog, labels)
    arrays = dict(schema_version=np.array(CATALOG_SCHEMA_VERSION), catalog=catalog)
    if labels is not None:
        arrays['labels'] = labels
    output = Path(path)
    with output.open('wb') as stream:
        (np.savez_compressed if compressed else np.savez)(stream, **arrays)
    return output


def load_shock_catalog(path, *, return_labels=False):
    """Load numeric catalog; optionally return stored original-record labels.

    Old object-catalog archives are rejected explicitly, not reinterpreted.
    """
    with np.load(path, allow_pickle=False) as archive:
        if 'schema_version' not in archive or int(archive['schema_version']) != CATALOG_SCHEMA_VERSION:
            raise ValueError('unsupported catalog schema; rebuild fronts from saved detections')
        catalog = archive['catalog']
        labels = archive['labels'] if 'labels' in archive else None
    _validate_catalog(catalog, labels)
    if return_labels:
        if labels is None:
            raise ValueError('archive has no membership labels')
        return catalog, labels
    return catalog


def save_shock_catalog_csv(path, catalog):
    """Write numeric front summaries; membership belongs in the NPZ archive."""
    catalog = _validate_catalog(catalog)
    fields = [name if name not in ('center', 'normal', 'extent') else f'{name}_{axis}'
              for name in front_dtype.names
              for axis in (('x', 'y', 'z') if name in ('center', 'normal', 'extent') else ('',))]
    output = Path(path)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in catalog:
            writer.writerow([value for name in front_dtype.names for value in np.atleast_1d(row[name])])
    return output
