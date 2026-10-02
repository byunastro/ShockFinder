"""Spatial fronts from saved shock-center detections, without origin inference."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .compact import ShockSamples

# Every flag describes the returned measurements or the accepted connectivity.
QUALITY_OK = 0
QUALITY_NO_AREA = 1 << 0             # Complete effective-area total unavailable.
QUALITY_NO_DISS_RATE = 1 << 1        # Complete cell-integrated power unavailable.
QUALITY_UNDEFINED_NORMAL = 1 << 2   # No usable normals, or their mean cancels.
QUALITY_APPROX_CONNECTIVITY = 1 << 3 # Adjacency inferred from positions/widths.
QUALITY_GAP_BRIDGED = 1 << 4         # At least one accepted link crosses a cell gap.
QUALITY_PARTIAL_SUMMARY = 1 << 5     # Missing/overflowed aggregate values.

front_dtype = np.dtype([
    ('front_id', '<i4'), ('ncell', '<i4'), ('center', '<f8', (3,)),
    ('normal', '<f4', (3,)), ('extent', '<f4', (3,)), ('area', '<f8'),
    ('mach', '<f4'), ('diss_rate', '<f8'), ('quality', '<u2'),
], align=False)

_LENGTH_TO_KPC = {'km': 1. / 3.0856775814913673e16, 'kpc': 1., 'Mpc': 1000.}
_I32_MAX = np.iinfo(np.int32).max


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _array(obj, name, shape, *, optional=False):
    value = _value(obj, name)
    if value is None and optional:
        return None
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {array.shape}')
    return array


def _ids(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or (values.size and values.min() < 0):
        raise ValueError(f'{name} must be a one-dimensional nonnegative integer array')
    if values.size and int(values.max()) > np.iinfo(np.int64).max:
        raise OverflowError(f'{name} exceeds int64')
    # Signed/unsigned int64 mixed searchsorted can promote IDs to float64.
    # Normalize unsigned IDs after the range check to preserve IDs above 2**53.
    return values.astype(np.int64) if values.dtype.kind == 'u' else values


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def shock_front_catalog(result, dissipation=None, *, return_labels=False,
                        min_group_size=3, mach_tolerance=0.3, normal_cosine=0.5,
                        surface_offset_factor=0.25, surface_angle_cosine=0.5,
                        connectivity='touch', gap_factor=0., min_mach=1.,
                        require_mach_consistent=False,
                        atol=1.e-9, rtol=1.e-7, normal_tolerance=1.e-6,
                        chunk_size=131072, query_chunk_size=4096,
                        max_neighbor_pairs=200000, thread=1):
    """Group saved ShockFinder centers in one cropped, open-boundary snapshot.

    result is ShockResult or ShockSamples (or a dense matching field mapping).
    Positions and dx require position_unit in km/kpc/Mpc; dx is mandatory.
    Only accepted unique shock centers contribute, never shock-zone aliases.
    Saved validity/endpoint flags exclude invalid detections. Finite Mach <=
    min_mach is excluded. Missing Mach retains records but cannot provide an edge.
    Missing normals skip unavailable normal tests and flag partial summaries.
    No cluster metadata, periodic wrapping, detector runs or file I/O is used.

    dissipation.area is effective center area in kpc2; dissipation.total is
    integrated power in erg/s. Flux is NEVER substituted for total. The official
    output contract is identical row count/order to result, including non-shock
    rows: ID-less arrays are indexed directly with no join or extra ID storage.
    Do not independently reorder/filter dissipation or pair different runs.
    Length checks cannot detect same-length mismatched files. Historical files
    carrying selected_indices still use ID joining, including reordered records.
    None uses ShockSamples embedded values, otherwise totals are NaN.
    Inputs are never modified.

    Options
    -------
    return_labels=False: also return int32 labels in original result-record
        order; -1 means excluded/undersized, otherwise snapshot-local front_id.
    min_group_size=3: minimum accepted center count, not a quality-only flag.
    mach_tolerance=0.3: maximum |Mi-Mj| / max(Mi,Mj) on each linked pair,
        in [0,1]. 0 requires equal Mach; 1 allows any finite eligible pair.
    normal_cosine=0.5: minimum signed unit-normal dot product, in [0,1].
    surface_offset_factor=0.25, surface_angle_cosine=0.5: displacement along
        EACH normal must be <= factor * smaller dx AND <= cosine * distance.
        Local tests permit gradual curvature without a global mean-normal cut.
    connectivity='touch': cube face/edge/corner contact; 'face' restricts to
        positive face overlap. AMR dx determines contact; no mesh rebuild.
    gap_factor=0: disabled. Positive values extend reach by larger dx * factor,
        still subject to Mach/normal/tangential checks; accepted gaps flag 16.
    min_mach=1: lower Mach threshold (exclusive for finite values).
    require_mach_consistent=False: opt in to saved consistency diagnostics.
    atol=1e-9 (kpc), rtol=1e-7: absolute/relative contact tolerances.
    normal_tolerance=1e-6: weighted normal resultant threshold; cancellation
        returns NaN normal, retaining the front.
    thread=1: maximum spatial query threads; no child processes or main guard.
    chunk_size=131072: input selection and summary batch size.
    query_chunk_size=4096: spatial query batch size.
    max_neighbor_pairs=200000: streamed candidate edge batch limit.
        Batch limits do not cap total memory occupied by input/selected arrays.

    Returns
    -------
    front_dtype structured array, 78 bytes per row, no object fields:
    front_id int32 (in increasing minimum original input-cell ID), ncell int32;
    center float64[3] and extent float32[3] in kpc (extent includes dx);
    normal float32[3], mach float32 dimensionless; area float64 kpc2;
    diss_rate float64 erg/s; quality uint16. Integer overflow is checked.
    Center uses area weights only when ALL member areas are reliable, otherwise
    equal weights over all members. Mach/normal means use valid subsets with
    those weights. Incomplete area/rate totals are NaN, never partial totals.
    With return_labels=True return (catalog, labels); labels address original
    result rows, so labels == front_id retrieves that front's detections.

    Quality bits (OR-combined, QUALITY_OK=0):
    NO_AREA=1: complete area total unavailable/invalid/overflowed.
    NO_DISS_RATE=2: complete integrated-rate total unavailable/invalid/overflowed.
    UNDEFINED_NORMAL=4: missing or cancelling weighted mean normal.
    APPROX_CONNECTIVITY=8: adjacency inferred from cell positions and widths.
    GAP_BRIDGED=16: accepted link crosses a non-contact gap.
    PARTIAL_SUMMARY=32: some members lack valid aggregate values or a summary
        overflows. Valid-subset means carry this bit; incomplete totals are NaN.
    """
    min_group_size = _positive_int(min_group_size, 'min_group_size')
    for name, val in [('chunk_size', chunk_size), ('query_chunk_size', query_chunk_size),
                      ('max_neighbor_pairs', max_neighbor_pairs), ('thread', thread)]:
        _positive_int(val, name)
    if min_group_size > _I32_MAX:
        raise OverflowError('min_group_size exceeds int32 ncell')
    for name, val in [('return_labels', return_labels),
                      ('require_mach_consistent', require_mach_consistent)]:
        if not isinstance(val, (bool, np.bool_)):
            raise ValueError(f'{name} must be boolean')
    for name, val in [('mach_tolerance', mach_tolerance), ('normal_cosine', normal_cosine), ('surface_angle_cosine', surface_angle_cosine)]:
        if not np.isfinite(val) or not 0 <= val <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    for name, val in [('surface_offset_factor', surface_offset_factor), ('gap_factor', gap_factor),
                      ('atol', atol), ('rtol', rtol)]:
        if not np.isfinite(val) or val < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(min_mach) or min_mach < 1:
        raise ValueError('min_mach must be finite and >= 1')
    if not np.isfinite(normal_tolerance) or not 0 <= normal_tolerance < 1:
        raise ValueError('normal_tolerance must be in [0, 1)')
    if connectivity not in ('touch', 'face'):
        raise ValueError("connectivity must be 'touch' or 'face'")
    data, nrecords = _prepare(result, dissipation, min_mach,
                              require_mach_consistent, chunk_size)
    n = len(data['rows'])
    labels = np.full(nrecords, -1, np.int32) if return_labels else None
    if not n:
        empty = np.empty(0, dtype=front_dtype)
        return (empty, labels) if return_labels else empty
    components = _Components(n)
    flags = np.full(n, QUALITY_APPROX_CONNECTIVITY, np.uint16)
    options = (normal_cosine, surface_offset_factor, surface_angle_cosine,
               connectivity, gap_factor, atol, rtol, mach_tolerance)
    for left, right in _spatial_pairs(data, gap_factor, atol, rtol,
                                      query_chunk_size, max_neighbor_pairs, thread):
        _accept(left, right, data, components, flags, options)
    roots = components.roots(np.arange(n))
    order = np.argsort(roots, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(roots[order]))+1, n]
    counts = np.diff(cuts)
    keep = np.flatnonzero(counts >= min_group_size)
    if len(keep) > _I32_MAX or (counts.size and counts.max() > _I32_MAX):
        raise OverflowError('front_id or ncell exceeds int32')
    catalog = np.empty(len(keep), dtype=front_dtype)
    for front_id, group in enumerate(keep):
        members = order[cuts[group]:cuts[group+1]]
        quality = int(np.bitwise_or.reduce(flags[members]))
        catalog[front_id] = _summarize(front_id, members, data, quality, normal_tolerance, chunk_size)
        if labels is not None:
            labels[data['rows'][members]] = front_id
    return (catalog, labels) if return_labels else catalog


def _prepare(result, dissipation, min_mach, consistent, chunk):
    compact = isinstance(result, ShockSamples)
    source = result.columns if compact else result
    pos = np.asarray(_value(source, 'pos'))
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.dtype.kind not in 'fiu':
        raise ValueError('pos must be a real numeric (N, 3) array')
    n = len(pos)
    mach = _array(source, 'mach', (n,), optional=True)
    unit = result.metadata.get('position_unit') if compact else _value(source, 'position_unit')
    if unit not in _LENGTH_TO_KPC:
        raise ValueError('position_unit must explicitly be km, kpc, or Mpc')
    factor = _LENGTH_TO_KPC[unit]
    ids = _ids(_array(source, 'input_row' if compact else 'selected_indices', (n,)), 'result IDs')
    shock = _array(source, 'shock', (n,), optional=compact)
    if shock is not None and shock.dtype.kind != 'b':
        raise ValueError('shock must be a boolean mask')
    centers = _array(source, 'center_index', (n,), optional=compact)
    if centers is not None and centers.dtype.kind not in 'iu':
        raise ValueError('center_index must contain integer retained-row indices')
    dx = _array(source, 'dx', (n,))
    normal = _array(source, 'normal', (n, 3), optional=True)
    valid_mask = _array(source, 'valid', (n,), optional=True)
    if valid_mask is not None and valid_mask.dtype.kind != 'b':
        raise ValueError('valid must be a boolean mask')
    status = _array(source, 'mach_validation_status', (n,), optional=True)
    if status is not None and status.dtype.kind not in 'iu':
        raise ValueError('mach_validation_status must be integer')
    mach_ok = _array(source, 'mach_consistent', (n,), optional=True)
    if mach_ok is not None and mach_ok.dtype.kind != 'b':
        raise ValueError('mach_consistent must be boolean')
    if consistent and mach_ok is None and status is None:
        raise ValueError('require_mach_consistent requires saved validation information')
    endpoints = [_array(source, key, (n,), optional=True) for key in ('upstream_index', 'downstream_index')]
    if any(endpoint is not None and endpoint.dtype.kind not in 'iu' for endpoint in endpoints):
        raise ValueError('saved endpoint indices must be integer')
    def selected_blocks():
        for start in range(0, n, chunk):
            stop = min(start+chunk, n)
            sl = slice(start, stop)
            rows = np.arange(start, stop, dtype=np.int64)
            valid = np.ones(stop-start, bool) if shock is None else shock[sl].copy()
            if centers is not None:
                valid &= centers[sl] == (ids[sl] if compact else rows)
            valid &= np.all(np.isfinite(pos[sl]), axis=1)
            if mach is not None:
                valid &= ~np.isfinite(mach[sl]) | (mach[sl] > min_mach)
            if valid_mask is not None:
                valid &= valid_mask[sl]
            if status is not None:
                valid &= (status[sl] & (1 << 8)) == 0
            for endpoint in endpoints:
                if endpoint is not None:
                    valid &= endpoint[sl] >= 0
                    if not compact:
                        valid &= endpoint[sl] < n
            if consistent:
                valid &= mach_ok[sl].astype(bool) if mach_ok is not None else (status[sl] & (1 << 7)) != 0
            if dx is not None:
                valid &= np.isfinite(dx[sl]) & (dx[sl] > 0)
            yield rows[valid]
    # Two passes avoid a list of per-chunk arrays and an extra full concatenation.
    count = sum(len(rows) for rows in selected_blocks())
    rows = np.empty(count, np.int64)
    offset = 0
    for block in selected_blocks():
        rows[offset:offset+len(block)] = block
        offset += len(block)
    cell_ids = ids[rows]
    order = np.argsort(cell_ids, kind='stable')
    if count and np.any(cell_ids[order][1:] == cell_ids[order][:-1]):
        raise ValueError('duplicate contributing shock-center input IDs')
    rows = rows[order]
    normals = np.full((count, 3), np.nan) if normal is None else np.asarray(normal[rows], dtype=float)
    norm = np.linalg.norm(normals, axis=1)
    normal_valid = np.all(np.isfinite(normals), axis=1) & np.isfinite(norm) & (norm > 0)
    normals[normal_valid] /= norm[normal_valid, None]
    widths = np.asarray(dx[rows], dtype=float)*factor
    if not np.all(np.isfinite(widths) & (widths > 0)):
        raise ValueError('converted cell widths must be finite and positive')
    positions = np.asarray(pos[rows], dtype=np.float64)*factor
    if not np.all(np.isfinite(positions)):
        raise ValueError('converted positions must be finite')
    area, total = _dissipation(source, ids, rows, dissipation, compact, chunk)
    return dict(rows=rows, pos=positions, dx=widths,
                normal=normals, normal_valid=normal_valid,
                mach=np.full(count, np.nan) if mach is None else np.asarray(mach[rows], dtype=float),
                area=area, total=total), n


def _dissipation(source, ids, rows, diss, compact, chunk):
    if diss is None:
        if compact:
            return tuple(np.full(len(rows), np.nan) if _value(source, 'dissipation_'+key) is None
                         else np.asarray(_array(source, 'dissipation_'+key, (len(ids),))[rows], dtype=float)
                         for key in ('area', 'total'))
        return np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for key, expected in [('area_unit', 'kpc2'), ('total_unit', 'erg/s')]:
        if _value(diss, key, expected) != expected:
            raise ValueError(f'{key} must be {expected}; convert saved values explicitly')
    diss_ids = _value(diss, 'selected_indices')
    if diss_ids is None:
        n = len(ids)
        for key in ('flux', 'efficiency', 'sound_speed'):
            _array(diss, key, (n,), optional=True)
        output = []
        for key in ('area', 'total'):
            values = _array(diss, key, (n,), optional=True)
            output.append(np.full(len(rows), np.nan) if values is None
                          else np.asarray(values[rows], dtype=float))
        return tuple(output)
    else:
        diss_ids = _ids(diss_ids, 'dissipation.selected_indices')
        n = len(diss_ids)
        same = n == len(ids) and all(np.array_equal(diss_ids[s:s+chunk], ids[s:s+chunk])
                                     for s in range(0, n, chunk))
        # Validate uniqueness even on the fast aligned path; monotonic IDs
        # avoid sorting an entire multi-GB retained-cell table.
        monotonic = all(np.all(diss_ids[max(0, s-1):s+chunk][1:] > diss_ids[max(0, s-1):s+chunk][:-1])
                        for s in range(0, n, chunk))
        order = None if monotonic else np.argsort(diss_ids, kind='stable')
        sorted_ids = diss_ids if order is None else diss_ids[order]
        if not monotonic and np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise ValueError('duplicate dissipation IDs')
        if same:
            index, found = rows, np.ones(len(rows), bool)
        else:
            loc = np.searchsorted(sorted_ids, ids[rows])
            found = loc < n
            found[found] &= sorted_ids[loc[found]] == ids[rows[found]]
            index = np.zeros(len(rows), np.int64)
            index[found] = loc[found] if order is None else order[loc[found]]
    # Validate any additional supplied columns, without copying them.
    for key in ('flux', 'efficiency', 'sound_speed'):
        _array(diss, key, (n,), optional=True)
    out = []
    for key in ('area', 'total'):
        values = _array(diss, key, (n,), optional=True)
        field = np.full(len(rows), np.nan)
        if values is not None:
            field[found] = values[index[found]]
        out.append(field)
    return tuple(out)


class _Components:
    """Stream sparse links into a forest without retaining the full graph."""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32 if n <= _I32_MAX else np.int64)
        self.kernel = None
        if n <= _I32_MAX:
            try:
                from . import _merger_neighbors
                kernel = getattr(_merger_neighbors, 'merger_neighbor_kernel', None)
                self.kernel = getattr(kernel, 'merge_component_labels', None)
            except ImportError:
                pass

    def roots(self, indices):
        roots = self.parent[indices]
        while not np.array_equal(roots, self.parent[roots]):
            roots = self.parent[roots]
        self.parent[indices] = roots
        return roots

    def merge(self, left, right):
        if not len(left):
            return
        if self.kernel is not None:
            self.kernel(left.astype(np.int32), right.astype(np.int32), self.parent)
        else:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            # Reuse the previous streamed component reduction: the sparse graph
            # contains only roots touched by this batch, not every detection.
            left, right = self.roots(left), self.roots(right)
            unique, inverse = np.unique(np.r_[left, right], return_inverse=True)
            m = len(left)
            graph = coo_matrix((np.ones(m, bool), (inverse[:m], inverse[m:])),
                               shape=(len(unique), len(unique))).tocsr()
            count, labels = connected_components(graph, directed=False)
            minimum = np.full(count, np.iinfo(self.parent.dtype).max, self.parent.dtype)
            np.minimum.at(minimum, labels, unique)
            self.parent[unique] = minimum[labels]


def _spatial_pairs(data, gap, atol, rtol, query_chunk, budget, thread):
    from scipy.spatial import cKDTree
    widths = data['dx']
    levels = np.floor(np.log2(widths)-np.log2(widths.min())).astype(np.int64)
    buckets = [np.flatnonzero(levels == level) for level in np.unique(levels)]
    for b, targets in enumerate(buckets):
        tree = cKDTree(data['pos'][targets])
        largest = widths[targets].max()
        for a, sources in enumerate(buckets[:b+1]):
            for start in range(0, len(sources), query_chunk):
                src = sources[start:start+query_chunk]
                radii = (.5*(widths[src]+largest)+gap*np.maximum(widths[src], largest))
                radii = radii+atol+rtol*radii
                for left, right in _tree_pairs(tree, targets, src, data['pos'][src], radii, budget, thread):
                    if a == b:
                        use = left < right
                        left, right = left[use], right[use]
                    yield left, right


def _tree_pairs(tree, targets, sources, points, radii, budget, thread):
    """Bound neighbor lists even when one source has more candidates than budget."""
    counts = tree.query_ball_point(points, radii, p=np.inf, return_length=True, workers=min(thread, len(points)))
    cumulative = np.r_[0, np.cumsum(counts)]
    lo = 0
    while lo < len(sources):
        if counts[lo] > budget:
            for start in range(0, len(targets), budget):
                right = targets[start:start+budget]
                yield np.full(len(right), sources[lo], dtype=sources.dtype), right
            lo += 1
            continue
        hi = min(max(lo+1, int(np.searchsorted(cumulative, cumulative[lo]+budget, side='right')-1)), len(sources))
        neighborhoods = tree.query_ball_point(points[lo:hi], radii[lo:hi], p=np.inf, workers=min(thread, hi-lo))
        if np.sum(counts[lo:hi]):
            yield np.repeat(sources[lo:hi], counts[lo:hi]), targets[np.concatenate(neighborhoods).astype(np.int64)]
        lo = hi


def _accept(left, right, data, components, flags, options):
    cosine, offset_factor, angle, connectivity, gap, atol, rtol, mach_tolerance = options
    use = left != right
    left, right = left[use], right[use]
    if not len(left):
        return
    delta = data['pos'][right]-data['pos'][left]
    distance = np.linalg.norm(delta, axis=1)
    small = np.minimum(data['dx'][left], data['dx'][right])
    half = .5*(data['dx'][left]+data['dx'][right])
    reach = half+gap*np.maximum(data['dx'][left], data['dx'][right])
    tolerance = atol+rtol*reach
    touch = np.all(np.abs(delta) <= (half+tolerance)[:, None], axis=1)
    close = np.all(np.abs(delta) <= (reach+tolerance)[:, None], axis=1)
    if connectivity == 'face':
        interior = np.abs(delta) < (half-tolerance)[:, None]
        boundary = np.abs(np.abs(delta)-half[:, None]) <= tolerance[:, None]
        touch &= (interior.sum(axis=1) == 2) & np.any(boundary, axis=1)
        close &= touch | ((gap > 0) & (interior.sum(axis=1) == 2))
    nv = data['normal_valid']
    both = nv[left] & nv[right]
    alignment = np.einsum('ij,ij->i', data['normal'][left], data['normal'][right])
    close &= ~both | (alignment >= cosine)
    ml, mr = data['mach'][left], data['mach'][right]
    finite_mach = np.isfinite(ml) & np.isfinite(mr)
    with np.errstate(invalid='ignore', divide='ignore'):
        relative = np.abs(ml-mr)/np.maximum(ml, mr)
    close &= finite_mach & (relative <= mach_tolerance)
    for side in (left, right):
        projection = np.abs(np.einsum('ij,ij->i', delta, data['normal'][side]))
        close &= ~nv[side] | ((projection <= offset_factor*small+tolerance)
                             & (projection <= angle*distance+tolerance))
    left, right = left[close], right[close]
    bridged = ~touch[close]
    flags[left[bridged]] |= QUALITY_GAP_BRIDGED | QUALITY_APPROX_CONNECTIVITY
    flags[right[bridged]] |= QUALITY_GAP_BRIDGED | QUALITY_APPROX_CONNECTIVITY
    components.merge(left, right)


def _summarize(front_id, rows, data, quality, normal_tol, chunk_size):
    """Two bounded passes; no full front-sized geometry/weight copies."""
    n = len(rows)
    if n > _I32_MAX:
        raise OverflowError('ncell exceeds int32')
    area_sum = power = mach_area = normal_area = 0.
    complete_area = complete_rate = True
    nmach = nnormal = 0
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, n, chunk_size):
            part = rows[start:start+chunk_size]
            area, rate = data['area'][part], data['total'][part]
            va = np.isfinite(area) & (area > 0)
            vr = np.isfinite(rate) & (rate >= 0)
            vm, vn = np.isfinite(data['mach'][part]), data['normal_valid'][part]
            complete_area &= bool(np.all(va))
            complete_rate &= bool(np.all(vr))
            area_sum += area[va].sum(dtype=float)
            power += rate[vr].sum(dtype=float)
            nmach += int(vm.sum())
            nnormal += int(vn.sum())
            mach_area += area[va & vm].sum(dtype=float)
            normal_area += area[va & vn].sum(dtype=float)
    reliable_area = complete_area and np.isfinite(area_sum) and area_sum > 0
    if not reliable_area:
        area_sum = np.nan
        quality |= QUALITY_NO_AREA | QUALITY_PARTIAL_SUMMARY
    if not complete_rate or not np.isfinite(power):
        power = np.nan
        quality |= QUALITY_NO_DISS_RATE | QUALITY_PARTIAL_SUMMARY
    if nmach != n or nnormal != n:
        quality |= QUALITY_PARTIAL_SUMMARY

    anchor = data['pos'][rows[0]].copy()
    center_delta, normal_sum = np.zeros(3), np.zeros(3)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    mean_mach = 0. if nmach else np.nan
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, n, chunk_size):
            part = rows[start:start+chunk_size]
            pos = data['pos'][part].copy()
            pos -= anchor
            area = data['area'][part]
            weights = area/area_sum if reliable_area else np.full(len(part), 1./n)
            center_delta += np.sum(pos*weights[:, None], axis=0)
            half = data['dx'][part, None]*.5
            lower = np.minimum(lower, np.min(pos-half, axis=0))
            upper = np.maximum(upper, np.max(pos+half, axis=0))
            mach = data['mach'][part]
            vm, vn = np.isfinite(mach), data['normal_valid'][part]
            if np.any(vm):
                weights_m = area[vm]/mach_area if reliable_area else np.full(int(vm.sum()), 1./nmach)
                mean_mach += np.sum(mach[vm]*weights_m)
            if np.any(vn):
                weights_n = area[vn]/normal_area if reliable_area else np.full(int(vn.sum()), 1./nnormal)
                normal_sum += np.sum(data['normal'][part[vn]]*weights_n[:, None], axis=0)
        center, extent = anchor+center_delta, upper-lower
    normal = np.full(3, np.nan)
    norm = np.linalg.norm(normal_sum)
    if nnormal and np.isfinite(norm) and norm > normal_tol:
        normal = normal_sum/norm
    else:
        quality |= QUALITY_UNDEFINED_NORMAL
    if not np.all(np.isfinite(center)):
        center[:] = np.nan
        quality |= QUALITY_PARTIAL_SUMMARY
    if not np.all(np.isfinite(extent)) or np.any(extent > np.finfo(np.float32).max):
        extent[:] = np.nan
        quality |= QUALITY_PARTIAL_SUMMARY
    if not np.isfinite(mean_mach) or abs(mean_mach) > np.finfo(np.float32).max:
        mean_mach = np.nan
        quality |= QUALITY_PARTIAL_SUMMARY
    return front_id, n, center, normal, extent, area_sum, mean_mach, power, quality
