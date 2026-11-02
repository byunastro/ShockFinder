"""Independent, shock-only products for retaining many snapshots in memory."""
from dataclasses import dataclass, fields
import json
from pathlib import Path
import numpy as np


@dataclass(slots=True)
class ShockSamples:
    """Columns contain detected centers only; row references remain explicit.

    ``retained_row`` addresses the original dense result; ``input_row`` addresses
    the input cell table. Endpoint indices are input-table indices, not sample
    indices. For interpolated endpoints, these identify containing cells, while
    endpoint positions and thermodynamic states are stored in separate columns.
    group_id is the front ID; groups stores the numeric front catalog. Arrays own
    their storage, so the dense analysis may be cleared immediately.
    """
    columns: dict[str, np.ndarray]
    groups: np.ndarray
    metadata: dict
    timings: dict

    def __len__(self):
        return len(self.columns['input_row'])

    def __getitem__(self, name):
        return self.columns[name]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.columns.values())

    @property
    def counts(self):
        return dict(self.metadata['counts'])

    def save(self, path, *, compressed=True):
        """Losslessly save columns, groups and metadata without pickle."""
        from .catalog_io import _validate_catalog
        if any(value.dtype.hasobject for value in self.columns.values()):
            raise ValueError('object columns cannot be saved without pickle')
        _validate_catalog(self.groups, self.columns.get('group_id'))
        manifest = dict(version=2, columns=list(self.columns), metadata=self.metadata,
                        timings=self.timings)
        arrays = {'manifest': np.asarray(json.dumps(manifest)), 'catalog': self.groups}
        arrays.update({'column/' + name: value for name, value in self.columns.items()})
        output = Path(path)
        writer = np.savez_compressed if compressed else np.savez
        with output.open('wb') as stream:
            writer(stream, **arrays)
        return output

    @classmethod
    def load(cls, path):
        """Restore the saved column values/dtypes and complete group summaries."""
        from .catalog_io import _validate_catalog
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive['manifest'].item()))
            if manifest.get('version') != 2:
                raise ValueError('unsupported compact archive version; regenerate from saved detections')
            columns = {name: archive['column/' + name] for name in manifest['columns']}
            if 'input_row' not in columns:
                raise ValueError('compact archive is missing input_row')
            n = len(columns['input_row'])
            if any(a.ndim == 0 or len(a) != n or a.dtype.hasobject for a in columns.values()):
                raise ValueError('invalid compact column shape or dtype')
            catalog = archive['catalog']
        _validate_catalog(catalog, columns.get('group_id'))
        return cls(columns, catalog, manifest['metadata'], manifest['timings'])

    def clear(self):
        self.columns.clear()
        self.groups = self.groups[:0].copy()
        self.metadata.clear()
        self.timings.clear()


_SCIENCE_COLUMNS = frozenset({
    'retained_row', 'input_row', 'mach', 'pos', 'dx', 'normal', 'level', 'zone_width',
    'upstream_index', 'downstream_index', 'mach_consistent', 'mach_validation_status',
    'dissipation_flux', 'dissipation_total', 'dissipation_area', 'group_id',
    'representative_input_row',
    'upstream_temperature', 'downstream_temperature',
    'upstream_density', 'downstream_density', 'upstream_pressure', 'downstream_pressure',
    'upstream_distance', 'downstream_distance', 'upstream_pos', 'downstream_pos',
    'sampling_converged', 'relative_ratio_change',
    'upstream_relative_temperature_change', 'downstream_relative_temperature_change',
})


def _sampled_states(result, rows, names=('temperature', 'density', 'pressure', 'distance', 'pos')):
    """Join sparse sampled states to dense result rows, preserving their order.

    A sampled endpoint's index identifies its containing cell; reading that
    cell cannot reconstruct the interpolated thermodynamic state or position.
    Missing measured rows therefore raise instead of silently changing states.
    Dictionaries are accepted for the saved galaxy-analysis interfaces.
    """
    sampling = result.get('sampling') if isinstance(result, dict) else getattr(result, 'sampling', None)
    if sampling is None:
        return None
    sampled_rows = np.asarray(sampling['rows'])
    rows = np.asarray(rows)
    if (sampled_rows.ndim != 1 or sampled_rows.dtype.kind not in 'iu'
            or np.any(sampled_rows[1:] <= sampled_rows[:-1])):
        raise ValueError('sampling rows must contain sorted unique result-row indices')
    indices = np.searchsorted(sampled_rows, rows)
    if (np.any(indices >= len(sampled_rows))
            or np.any(sampled_rows[indices] != rows)):
        raise ValueError('sampling is missing a requested shock result row')
    states = {}
    for name in names:
        values = np.asarray(sampling[name])
        shape = (len(sampled_rows), 2, 3) if name == 'pos' else (len(sampled_rows), 2)
        if values.shape != shape:
            raise ValueError(f'sampling {name} must have shape {shape}')
        states[name] = values[indices]
    return states


def compact_shocks(result, dissipation=None, catalog=None, *, labels=None, timings=None,
                   profile='full', index_dtype='int64', _validation=None,
                   _compact_dissipation=False):
    """Compact without changing float precision or scientific selection.

    full preserves existing columns. science omits redundant and diagnostic
    columns while retaining primary physics, endpoint indices and quality flags.
    auto narrows index columns only when all their integer values fit int32.
    """
    if profile not in ('full', 'science'):
        raise ValueError("profile must be 'full' or 'science'")
    if index_dtype not in ('int64', 'auto'):
        raise ValueError("index_dtype must be 'int64' or 'auto'")
    keep = (lambda name: True) if profile == 'full' else _SCIENCE_COLUMNS.__contains__
    rows = np.flatnonzero(result.shock)
    columns = {'retained_row': rows, 'input_row': result.selected_indices[rows]}
    excluded = {'selected_indices', 'center_index', 'upstream_index', 'downstream_index', 'mach_temperature'}
    for field in fields(result):
        value = getattr(result, field.name)
        if field.name not in excluded and keep(field.name) and isinstance(value, np.ndarray):
            columns[field.name] = value[rows]
    if _validation is not None:
        for field in fields(_validation):
            value = getattr(_validation, field.name)
            if (field.name not in excluded and field.name not in ('mach', 'shock')
                    and keep(field.name) and isinstance(value, np.ndarray)):
                columns[field.name] = value[_validation.shock]
    for name in ('center_index', 'upstream_index', 'downstream_index'):
        if not keep(name):
            continue
        indices = getattr(result, name)[rows]
        valid = (indices >= 0) & (indices < result.mach.size)
        mapped = np.full(len(rows), -1, dtype=np.int64)
        mapped[valid] = result.selected_indices[indices[valid]]
        columns[name] = mapped
        if profile == 'full' and name != 'center_index' and result.pos is not None:
            pos = np.full((len(rows), 3), np.nan)
            pos[valid] = result.pos[indices[valid]]
            columns[name.replace('_index', '_pos')] = pos
    sampled = _sampled_states(result, rows)
    if sampled is not None:
        for name, values in sampled.items():
            for side, endpoint in enumerate(('upstream', 'downstream')):
                columns[endpoint + '_' + name] = values[:, side].copy()
        raw_sampling = result.sampling
        slots = np.searchsorted(raw_sampling['rows'], rows)
        if 'converged' in raw_sampling:
            columns['sampling_converged'] = raw_sampling['converged'][slots].copy()
        if 'relative_ratio_change' in raw_sampling:
            columns['relative_ratio_change'] = raw_sampling['relative_ratio_change'][slots].copy()
        if 'relative_temperature_change' in raw_sampling:
            for side, endpoint in enumerate(('upstream', 'downstream')):
                columns[endpoint + '_relative_temperature_change'] = (
                    raw_sampling['relative_temperature_change'][slots, side].copy())
    if dissipation is not None:
        for field in fields(dissipation):
            name = 'dissipation_' + field.name
            if keep(name):
                value = getattr(dissipation, field.name)
                columns[name] = value.copy() if _compact_dissipation else value[rows]
    from .fronts import front_dtype
    groups = np.empty(0, dtype=front_dtype)
    representatives = len(rows)
    if catalog is not None:
        from .catalog_io import _validate_catalog
        if labels is None or np.shape(labels) != (result.mach.size,):
            raise ValueError('compact catalog requires labels aligned with result')
        _validate_catalog(catalog, labels)
        columns['group_id'] = labels[rows].copy()
        groups = catalog.copy()
        representatives = int(catalog['ncell'].sum(dtype=np.int64))
    if index_dtype == 'auto':
        for name, value in columns.items():
            if value.dtype == np.int64 and (not value.size or
                    (value.min() >= np.iinfo(np.int32).min and value.max() <= np.iinfo(np.int32).max)):
                columns[name] = value.astype(np.int32)
    metadata = {
        'profile': profile, 'index_dtype': index_dtype,
        'gamma': result.gamma, 'temperature_floor': result.temperature_floor,
        'position_unit': result.position_unit, 'endpoint_index_space': 'input_table',
        'group_index_space': 'snapshot_front_id',
        'counts': {'retained': result.mach.size, 'shock': len(rows),
                   'representative': representatives, 'groups': len(groups)},
        'diagnostics': dict(result.diagnostics or {}),
        'catalog': {'schema': 'front_dtype', 'bytes_per_row': 78} if catalog is not None else None,
    }
    if sampled is not None:
        metadata['endpoint_index_role'] = 'representative_containing_cell'
        metadata['endpoint_state_sampling'] = 'interpolated'
        if 'converged' in result.sampling:
            metadata['sampling_tolerance'] = float(result.sampling_tolerance)
            metadata['endpoint_selection'] = 'temperature_extrema_with_density_pressure_ordering'
        metadata['sampled_state_units'] = {
            'temperature': getattr(result, 'temperature_unit', 'K'),
            'density': getattr(result, 'density_unit', 'Msol/kpc3'),
            'pressure': (getattr(result, 'density_unit', 'Msol/kpc3') + ' '
                         + getattr(result, 'temperature_unit', 'K')),
            'distance': result.position_unit, 'pos': result.position_unit,
        }
    return ShockSamples(columns, groups, metadata, dict(timings or {}))
