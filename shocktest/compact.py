"""Independent, shock-only products for retaining many snapshots in memory."""
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import numpy as np


@dataclass(slots=True)
class ShockSamples:
    """Columns contain detected centers only; row references remain explicit.

    ``retained_row`` addresses the original dense result; ``input_row`` addresses
    the input cell table. Endpoint indices are input-table indices, not sample
    indices. Group center_indices also address that input table. Arrays own
    their storage, so the dense analysis may be cleared immediately.
    """
    columns: dict[str, np.ndarray]
    groups: tuple
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
        from .catalog import ShockCatalog
        from .catalog_io import _catalog_arrays
        if any(value.dtype.hasobject for value in self.columns.values()):
            raise ValueError('object columns cannot be saved without pickle')
        manifest = dict(version=1, columns=list(self.columns), metadata=self.metadata,
                        timings=self.timings)
        arrays = {'manifest': np.asarray(json.dumps(manifest))}
        arrays.update({'column/' + name: value for name, value in self.columns.items()})
        group_catalog = ShockCatalog(np.empty(0, np.int64), np.empty(0, np.int64),
                                     list(self.groups), {})
        arrays.update({'catalog/' + name: value for name, value in _catalog_arrays(group_catalog).items()})
        output = Path(path)
        writer = np.savez_compressed if compressed else np.savez
        with output.open('wb') as stream:
            writer(stream, **arrays)
        return output

    @classmethod
    def load(cls, path):
        """Restore the saved column values/dtypes and complete group summaries."""
        from .catalog_io import _catalog_from_arrays, _required_fields, CATALOG_SCHEMA_VERSION
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive['manifest'].item()))
            if manifest.get('version') != 1:
                raise ValueError('unsupported compact archive version')
            columns = {name: archive['column/' + name] for name in manifest['columns']}
            if 'input_row' not in columns:
                raise ValueError('compact archive is missing input_row')
            n = len(columns['input_row'])
            if any(a.ndim == 0 or len(a) != n or a.dtype.hasobject for a in columns.values()):
                raise ValueError('invalid compact column shape or dtype')
            version = int(archive['catalog/schema_version'].item())
            if version not in (1, CATALOG_SCHEMA_VERSION):
                raise ValueError('unsupported embedded catalog version')
            group_arrays = {name: archive['catalog/' + name] for name in _required_fields(version)}
        catalog = _catalog_from_arrays(group_arrays, version)
        return cls(columns, tuple(catalog.groups), manifest['metadata'], manifest['timings'])

    def clear(self):
        self.columns.clear()
        self.groups = ()
        self.metadata.clear()
        self.timings.clear()


_SCIENCE_COLUMNS = frozenset({
    'retained_row', 'input_row', 'mach', 'pos', 'dx', 'normal', 'level', 'zone_width',
    'upstream_index', 'downstream_index', 'mach_consistent', 'mach_validation_status',
    'dissipation_flux', 'dissipation_total', 'dissipation_area', 'group_id',
    'representative_input_row',
})


def compact_shocks(result, dissipation=None, catalog=None, *, timings=None,
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
    if dissipation is not None:
        for field in fields(dissipation):
            name = 'dissipation_' + field.name
            if keep(name):
                value = getattr(dissipation, field.name)
                columns[name] = value.copy() if _compact_dissipation else value[rows]
    groups = ()
    representatives = len(rows)
    if catalog is not None:
        columns['group_id'] = catalog.group_id[rows].copy()
        rep = catalog.center_representative[rows]
        valid = rep >= 0
        mapped = np.full(len(rows), -1, dtype=np.int64)
        mapped[valid] = result.selected_indices[rep[valid]]
        columns['representative_input_row'] = mapped
        representatives = int(np.count_nonzero(rep == rows))
        groups = tuple(replace(group, center_indices=result.selected_indices[group.center_indices].copy())
                       for group in catalog.groups)
    if index_dtype == 'auto':
        for name, value in columns.items():
            if value.dtype == np.int64 and (not value.size or
                    (value.min() >= np.iinfo(np.int32).min and value.max() <= np.iinfo(np.int32).max)):
                columns[name] = value.astype(np.int32)
    metadata = {
        'profile': profile, 'index_dtype': index_dtype,
        'gamma': result.gamma, 'temperature_floor': result.temperature_floor,
        'position_unit': result.position_unit, 'endpoint_index_space': 'input_table',
        'group_index_space': 'input_table',
        'counts': {'retained': result.mach.size, 'shock': len(rows),
                   'representative': representatives, 'groups': len(groups)},
        'diagnostics': dict(result.diagnostics or {}),
        'catalog': dict(catalog.metadata) if catalog is not None else None,
    }
    return ShockSamples(columns, groups, metadata, dict(timings or {}))
