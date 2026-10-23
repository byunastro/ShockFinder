"""Optional native H0 superlevel persistence and seeded maximum-spanning forest.

This is inspired by DisPerSE's persistence methodology, not a DisPerSE binding
or a discrete Morse complex. No compiler or subprocess runs during grouping.
"""
import ctypes
from pathlib import Path
import sys

import numpy as np


class FluxComponents:
    """Collect accepted edges, then partition their flux-weighted forest."""

    def __init__(self, flux, threshold):
        if len(flux) > np.iinfo(np.int32).max:
            raise OverflowError('persistence vertex count exceeds int32')
        if not np.all(np.isfinite(flux) & (flux > 0)):
            raise ValueError('flux_persistence requires finite positive flux for every selected center')
        suffix = '.dylib' if sys.platform == 'darwin' else '.so'
        path = Path(__file__).with_name('_flux_persistence' + suffix)
        try:
            self.lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise RuntimeError('Build the optional helper with python3 tools/build_persistence.py') from exc
        integer = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
        long = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
        real = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        self.lib.persistence.argtypes = [ctypes.c_int32, ctypes.c_int64,
                                        integer, long, real, real, integer, integer]
        self.lib.persistence.restype = ctypes.c_int64
        self.lib.watershed.argtypes = [ctypes.c_int32, ctypes.c_int64,
                                      integer, real, ctypes.c_double, integer]
        self.lib.watershed.restype = ctypes.c_int
        self.scalar = np.ascontiguousarray(np.log10(flux), dtype=np.float64)
        self.threshold = max(float(threshold), 1.e-12)
        self.parts = []
        self.gaps = []

    def record_gaps(self, left, right):
        if len(left):
            self.gaps.append(np.column_stack((left, right)).astype(np.int32))

    def update_gap_flags(self, flags, roots, bit):
        # A discarded cross-region bridge must not flag either final region.
        flags &= np.uint16(0xffff ^ bit)
        for edges in self.gaps:
            same = roots[edges[:, 0]] == roots[edges[:, 1]]
            flags[edges[same].ravel()] |= bit
        self.gaps.clear()

    def merge(self, left, right):
        # Preserve the geometry iterator's deterministic AMR orientation, also
        # used by the original comparison (fine-to-coarse across resolutions).
        self.parts.append(np.column_stack((left, right)).astype(np.int32))

    def roots(self, indices):
        n = len(self.scalar)
        edges = np.concatenate(self.parts) if self.parts else np.empty((0, 2), np.int32)
        self.parts.clear()
        weights = np.minimum(self.scalar[edges[:, 0]], self.scalar[edges[:, 1]])
        # Endpoint ordering resolves equal-height edges independently of batches.
        order = np.lexsort((edges[:, 1], edges[:, 0], -weights)).astype(np.int64)
        del weights
        death = np.empty(n, np.float64)
        elder = np.empty(n, np.int32)
        tree = np.empty((n, 2), np.int32)
        size = self.lib.persistence(n, len(edges), edges, order, self.scalar,
                                    death, elder, tree)
        if size < 0:
            raise MemoryError('native persistence allocation failed')
        del edges, order, elder
        persistence = self.scalar - death
        labels = np.empty(n, np.int32)
        status = self.lib.watershed(n, size, tree, persistence, self.threshold, labels)
        if status < 0:
            raise MemoryError('native watershed allocation failed')
        if np.any(labels < 0):
            raise RuntimeError('persistence watershed left unassigned vertices')
        return labels[indices]
