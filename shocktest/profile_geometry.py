"""Dyadic leaf lookup and conservative virtual-cell interpolation, in Python.

Coordinates are relative to a coarsest leaf corner. No unit-box, simulation
name, analytic shock position, or assumed spherical geometry is used.
"""

import itertools
import numpy as np
from . import _profile_geometry

KEY = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])


def keys(q):
    return np.ascontiguousarray(q, dtype=np.int64).view(KEY).reshape(-1)


class LeafGeometry:
    def __init__(self, pos, dx):
        self.pos = np.asarray(pos)
        self.dx = np.asarray(dx)
        if (
            self.pos.shape != (len(self.dx), 3)
            or not len(self.dx)
            or not np.isfinite(self.pos).all()
            or not np.isfinite(self.dx).all()
            or np.any(self.dx <= 0)
        ):
            raise ValueError(
                "Leaf positions and positive widths must be finite and nonempty"
            )
        self.origin = self.pos[np.argmax(dx)] - 0.5 * dx.max()
        widths = np.unique(dx)
        if not np.allclose(
            np.log2(widths / widths[0]),
            np.rint(np.log2(widths / widths[0])),
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("Profile reconstruction requires dyadic AMR leaf widths")
        self.groups = []
        for h in widths:
            ids = np.flatnonzero(dx == h)
            lattice = (self.pos[ids] - self.origin) / h - 0.5
            if not np.allclose(lattice, np.rint(lattice), atol=1e-6, rtol=0):
                raise ValueError("Leaf centers must lie on a common dyadic lattice")
            q = np.rint(lattice).astype(np.int64)
            k = keys(q)
            order = np.argsort(k)
            if len(np.unique(k)) != len(k):
                raise ValueError("Duplicate leaf cells")
            self.groups.append((h, k[order], ids[order]))

    def locate(self, points):
        points = np.asarray(points)
        out = np.full(len(points), -1, dtype=np.int64)
        finite = np.isfinite(points).all(axis=1)
        for h, k, ids in self.groups:
            use = np.flatnonzero((out < 0) & finite)
            if not len(use):
                break
            coordinate = (points[use] - self.origin) / h
            near = np.rint(coordinate)
            coordinate = np.where(abs(coordinate - near) < 2e-12, near, coordinate)
            q = np.floor(coordinate).astype(np.int64)
            wanted = keys(q)
            at = np.searchsorted(k, wanted)
            ok = at < len(k)
            ok[ok] &= k[at[ok]] == wanted[ok]
            out[use[ok]] = ids[at[ok]]
        return out

    def virtual(self, points, h, values, depth=0):
        ids = self.locate(points)
        out = np.full((len(ids), values.shape[1]), np.nan)
        good = ids >= 0
        direct = good.copy()
        direct[good] &= self.dx[ids[good]] >= h[good] * (1 - 1e-12)
        out[direct] = values[ids[direct]]
        refine = np.flatnonzero(good & ~direct)
        if len(refine):
            if depth > 30:
                raise ValueError("Invalid dyadic hierarchy")
            total = np.zeros((len(refine), values.shape[1]))
            for signs in itertools.product((-1, 1), repeat=3):
                total += (
                    self.virtual(
                        points[refine] + 0.25 * h[refine, None] * signs,
                        h[refine] / 2,
                        values,
                        depth + 1,
                    )
                    / 8
                )
            out[refine] = total
        return out

    def sample(self, points, h, values):
        points = np.asarray(points)
        h = np.broadcast_to(h, len(points))
        coordinate = (points - self.origin) / h[:, None] - 0.5
        nearest = np.rint(coordinate)
        coordinate = np.where(abs(coordinate - nearest) < 2e-12, nearest, coordinate)
        low = np.floor(coordinate)
        fraction = coordinate - low
        out = np.zeros((len(points), values.shape[1]))
        for bits in itertools.product((0, 1), repeat=3):
            weight = np.prod(np.where(bits, fraction, 1 - fraction), axis=1)
            use = np.flatnonzero(weight > 1e-14)
            if not len(use):
                continue
            node = self.origin + (low[use] + bits + 0.5) * h[use, None]
            out[use] += weight[use, None] * self.virtual(node, h[use], values)
        out[self.locate(points) < 0] = np.nan
        return out

    def derivatives(self, temperature, velocity, pressure, chunk=65536, rows=None):
        """Same face-center differences and four fine-face averages as native."""
        selected=np.arange(len(self.pos)) if rows is None else np.asarray(rows)
        n = len(selected)
        div = np.zeros(n)
        gt = np.zeros((n, 3))
        gp = np.zeros((n, 3))
        values = np.column_stack((temperature, pressure, velocity))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            p = self.pos[selected[start:stop]]
            h = self.dx[selected[start:stop]]
            for axis in range(3):
                transverse = [k for k in range(3) if k != axis]
                sides = []
                for sign in (-1, 1):
                    summed = np.zeros((len(p), 6))
                    counts = np.zeros(len(p))
                    for a, b in itertools.product((-1, 1), repeat=2):
                        probe = p.copy()
                        probe[:, axis] += sign * (0.5 + 1e-7) * h
                        probe[:, transverse[0]] += a * 0.25 * h
                        probe[:, transverse[1]] += b * 0.25 * h
                        ids = self.locate(probe)
                        ok = ids >= 0
                        summed[ok, 0] += self.pos[ids[ok], axis]
                        summed[ok, 1:] += values[ids[ok]]
                        counts[ok] += 1
                    sides.append(
                        np.divide(
                            summed,
                            counts[:, None],
                            out=np.full_like(summed, np.nan),
                            where=counts[:, None] > 0,
                        )
                    )
                minus, plus = sides
                distance = plus[:, 0] - minus[:, 0]
                ok = np.isfinite(distance) & (distance > 0)
                rows = np.arange(start, stop)[ok]
                gt[rows, axis] = (plus[ok, 1] - minus[ok, 1]) / distance[ok]
                gp[rows, axis] = (plus[ok, 2] - minus[ok, 2]) / distance[ok]
                div[rows] += (plus[ok, 3 + axis] - minus[ok, 3 + axis]) / distance[ok]
        return div, gt, gp

    def derivatives_at(self, temperature, velocity, pressure, rows):
        return self.derivatives(temperature, velocity, pressure, rows=rows)


class NativeLeafGeometry(LeafGeometry):
    """Same measurement rules, hashed cell lookup and OpenMP Fortran loops."""

    native = True

    def __init__(self, pos, dx):
        self.pos = np.asfortranarray(pos, dtype=np.float64)
        self.dx = np.asfortranarray(dx, dtype=np.float64)
        if (
            self.pos.shape != (len(dx), 3)
            or not len(dx)
            or not np.isfinite(self.pos).all()
            or not np.isfinite(dx).all()
            or np.any(dx <= 0)
        ):
            raise ValueError(
                "Leaf positions and positive widths must be finite and nonempty"
            )
        self.origin = self.pos[np.argmax(dx)] - 0.5 * self.dx.max()
        self.widths = np.unique(dx)
        exponents = np.log2(self.widths / self.widths[0])
        if not np.allclose(exponents, np.rint(exponents), atol=1e-9, rtol=0):
            raise ValueError("Profile reconstruction requires dyadic AMR leaf widths")
        self.levels = np.searchsorted(self.widths, self.dx).astype(np.int32) + 1
        lattice = (self.pos - self.origin) / self.dx[:, None] - 0.5
        if not np.allclose(lattice, np.rint(lattice), atol=1e-6, rtol=0):
            raise ValueError("Leaf centers must lie on a common dyadic lattice")
        if np.any(abs(lattice) > 9e18):
            raise ValueError("AMR coordinate index exceeds int64")
        self.q = np.asfortranarray(np.rint(lattice), dtype=np.int64)
        del lattice
        nt = 1
        while nt < 2 * len(dx):
            nt *= 2
        if nt >= 2**31:
            raise ValueError("Native geometry hash exceeds int32 address space")
        self.kernel = _profile_geometry.profile_geometry_kernel
        self.table, duplicate = self.kernel.build_index(self.q, self.levels, nt)
        if duplicate:
            raise ValueError("Duplicate leaf cells")

    def locate(self, points):
        p = np.asfortranarray(points, dtype=np.float64)
        return (
            self.kernel.locate_batch(
                p, self.origin, self.widths, self.q, self.levels, self.table
            ).astype(np.int64)
            - 1
        )

    def sample(self, points, h, values):
        p = np.asfortranarray(points, dtype=np.float64)
        scale = np.asfortranarray(np.broadcast_to(h, len(p)), dtype=np.float64)
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError("Sampling widths must be finite and positive")
        # Bound the temporary cache independently of input mesh size. Old native
        # binaries remain usable until rebuilt; they use the original kernel.
        cached = getattr(self.kernel, 'sample_batch_cached', None)
        if cached is not None and len(p) >= 256:
            values = np.asfortranarray(values, dtype=np.float64)
            # Preserve the f2py output layout: downstream reductions must keep
            # the same floating-point accumulation path as the original kernel.
            result = np.empty((len(p), values.shape[1]), order='F')
            for start in range(0, len(p), 32768):
                stop = min(start + 32768, len(p))
                result[start:stop], _, _ = cached(
                    p[start:stop], scale[start:stop], self.origin, self.widths,
                    self.dx, self.q, self.levels, self.table, values)
            return result
        return self.kernel.sample_batch(
            p,
            scale,
            self.origin,
            self.widths,
            self.dx,
            self.q,
            self.levels,
            self.table,
            np.asfortranarray(values, dtype=np.float64),
        )

    def derivatives(self, temperature, velocity, pressure, chunk=65536):
        return self.kernel.derivatives(
            self.pos,
            self.dx,
            temperature,
            np.asfortranarray(velocity),
            pressure,
            self.origin,
            self.widths,
            self.q,
            self.levels,
            self.table,
        )

    def derivatives_at(self, temperature, velocity, pressure, rows):
        if not hasattr(self.kernel,'derivatives_at'):
            # Existing binaries remain usable until rebuilt.
            return tuple(a[rows] for a in self.derivatives(temperature,velocity,pressure))
        return self.kernel.derivatives_at(self.pos,self.dx,temperature,
            np.asfortranarray(velocity),pressure,self.origin,self.widths,self.q,
            self.levels,self.table,np.asarray(rows,dtype=np.int32)+1)


def make_geometry(pos, dx):
    return (NativeLeafGeometry if _profile_geometry is not None else LeafGeometry)(
        pos, dx
    )
