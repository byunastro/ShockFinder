"""Spatial queries with bounded pairwise workspace and no SciPy requirement."""
import numpy as np


def _blocks(n, m, memory_budget_bytes):
    if memory_budget_bytes < 4096:
        raise ValueError('memory_budget_bytes must be at least 4096')
    # Conservative allowance for live vector/scalar/bool pairwise temporaries.
    pairs = max(1, int(memory_budget_bytes) // 512)
    a = min(max(n, 1), max(1, int(np.sqrt(pairs))))
    b = min(max(m, 1), max(1, pairs // a))
    for i in range(0, n, a):
        for j in range(0, m, b):
            yield slice(i, min(i+a, n)), slice(j, min(j+b, m))


def nearest_points(points, reference, *, memory_budget_bytes=32 * 1024**2):
    points, reference = np.asarray(points), np.asarray(reference)
    if not len(reference):
        raise ValueError('reference must not be empty')
    index = np.full(len(points), -1, dtype=np.int64)
    best = np.full(len(points), np.inf)
    for a, b in _blocks(len(points), len(reference), memory_budget_bytes):
        delta = points[a, None, :] - reference[None, b, :]
        d2 = np.einsum('ijk,ijk->ij', delta, delta)
        local = d2.argmin(axis=1)
        values = d2[np.arange(len(local)), local]
        use = values < best[a]
        best[a][use] = values[use]
        index[a][use] = b.start + local[use]
    return index, np.sqrt(best)


def connected_components(points, radius, *, memory_budget_bytes=32 * 1024**2, use_scipy=True):
    """Union neighbors as discovered, without storing a full adjacency graph."""
    points = np.asarray(points, dtype=float)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    parent = np.arange(len(points))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    def union(i, j):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    tree = None
    if use_scipy:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
        except ImportError:
            pass
    if tree is not None:
        # At most one row's neighbors reside in memory, rather than all edges.
        for i, point in enumerate(points):
            for j in tree.query_ball_point(point, radius):
                if j > i:
                    union(i, j)
    else:
        for a, b in _blocks(len(points), len(points), memory_budget_bytes):
            if b.stop <= a.start:
                continue
            delta = points[a, None, :] - points[None, b, :]
            d2 = np.einsum('ijk,ijk->ij', delta, delta)
            ii, jj = np.nonzero(d2 <= radius * radius)
            for i, j in zip(ii + a.start, jj + b.start):
                if j > i:
                    union(int(i), int(j))
    roots = np.array([root(i) for i in range(len(points))], dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1]


def nearest_segment_shock(p0, p1, catalog, *, search_radius, width_factor,
                          zone_width_factor, memory_budget_bytes=32 * 1024**2):
    """Search whole segments; prefer patches whose finite shock zone is hit.

    This exact tiled search trades quadratic compute for bounded workspace.
    Multiple moving surfaces and time integration are handled by exposure.py.
    """
    pos = catalog['pos']
    n = len(p0)
    index = np.full(n, -1, dtype=np.int64)
    best = np.full(n, np.inf)
    matched = np.zeros(n, bool)
    zone = catalog.get('zone_width', np.zeros(len(pos)))
    for a, b in _blocks(n, len(pos), memory_budget_bytes):
        start = p0[a, None, :] - pos[None, b, :]
        delta = (p1[a] - p0[a])[:, None, :]
        length2 = np.sum(delta * delta, axis=2)
        fraction = np.divide(-np.sum(start * delta, axis=2), length2,
                             out=np.zeros(start.shape[:2]), where=length2 > 0)
        fraction = np.clip(fraction, 0, 1)
        closest = start + fraction[..., None] * delta
        distance = np.sqrt(np.sum(closest * closest, axis=2))
        normal = catalog['normal'][None, b, :]
        s0 = np.sum(start * normal, axis=2)
        ds = np.sum(delta * normal, axis=2)
        crossing = np.divide(-s0, ds, out=fraction.copy(), where=ds != 0)
        crossing = np.clip(crossing, 0, 1)
        relative = start + crossing[..., None] * delta
        signed = s0 + crossing * ds
        tangent = relative - signed[..., None] * normal
        width = width_factor * catalog['dx'][b]
        half = zone_width_factor * zone[b] + width
        hit = ((np.abs(signed) <= half) & (np.sum(tangent*tangent, axis=2) <= width**2)
               & catalog['valid_normal'][b] & ((distance <= search_radius) | (distance <= half)))
        # Select the nearest matching patch, then the nearest patch if none hit.
        has_hit = np.any(hit, axis=1)
        cost = np.where(hit | ~has_hit[:, None], distance, np.inf)
        local = cost.argmin(axis=1)
        d = distance[np.arange(len(local)), local]
        use = (has_hit & ~matched[a]) | ((has_hit == matched[a]) & (d < best[a]))
        index[a][use] = b.start + local[use]
        best[a][use] = d[use]
        matched[a][use] = has_hit[use]
    return index, best


def moving_patch_blocks(p0, p1, q0, q1, radius, half_width, *, memory_budget_bytes):
    """Conservative swept-sphere broad phase, followed by bounded exact tests.

    SciPy uses O(number of patches) index storage. Without SciPy the tiled
    all-pairs fallback has the same results and bounded pairwise workspace.
    """
    if memory_budget_bytes < 4096:
        raise ValueError('memory_budget_bytes must be at least 4096')
    if len(q0) == 0 or len(p0) == 0:
        return
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        for a, b in _blocks(len(p0), len(q0), memory_budget_bytes):
            yield a, np.arange(b.start, b.stop)
        return
    middle = .5 * (q0 + q1)
    support = .5 * np.linalg.norm(q1-q0, axis=1) + np.hypot(radius, half_width)
    tree = cKDTree(middle)
    maximum_support = float(np.max(support))
    cap = max(1, int(memory_budget_bytes) // 512)
    for i in range(len(p0)):
        point = .5 * (p0[i]+p1[i])
        reach = .5 * np.linalg.norm(p1[i]-p0[i])
        candidates = tree.query_ball_point(point, reach+maximum_support)
        # Exact cylindrical intersection follows; the broad phase cannot
        # discard a real hit, including a moving shock past a static galaxy.
        for start in range(0, len(candidates), cap):
            rows = np.asarray(candidates[start:start+cap], dtype=np.int64)
            keep = np.linalg.norm(middle[rows]-point, axis=1) <= reach+support[rows]
            rows = rows[keep]
            if len(rows):
                yield slice(i, i+1), rows
