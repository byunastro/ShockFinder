"""Parallel profile reconstruction with an immutable, complete AMR background.

Only seed work is partitioned. Every worker can locate every retained leaf,
including leaves outside its assigned spatial block. Read-only NPY mappings
share physical file pages among processes on a node, with no fork/OpenMP mix.
The background is replicated across nodes, not across ranks on the same node.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import hashlib
import gzip
import shutil
import tempfile
import fcntl
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import socket
import time
import traceback
import uuid

import numpy as np

from . import __version__, _profile_geometry
from .core import ShockFinder
from .profile_geometry import NativeLeafGeometry
from . import profile_reconstruction as profile

SCHEMA = 1
_WORKER = None
_CONTEXT_ARRAYS = ('pos', 'dx', 'temperature', 'velocity', 'pressure',
                   'normals', 'known', 'values', 'seeds')
_GEOMETRY_ARRAYS = ('origin', 'widths', 'q', 'levels', 'table')


def _digest():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ('distributed.py', 'core.py', 'profile_quality.py',
                 'profile_reconstruction.py', '_profile_model.py',
                 'profile_geometry.py', 'fortran/profile_geometry.f90'):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _measurement_digest():
    names = ('core.py', 'profile_quality.py', 'profile_reconstruction.py',
             '_profile_model.py', 'profile_geometry.py', 'fortran/profile_geometry.f90')
    return hashlib.sha256(b''.join((Path(__file__).parent / n).read_bytes() for n in names)).hexdigest()


def _json_setting(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _finder(settings):
    finder = ShockFinder()
    for name, value in settings.items():
        if name == 'thermal_pressure_field' and isinstance(value, list):
            value = tuple(value)
        setattr(finder, name, value)
    finder.show_progress = False
    finder._validate_settings()
    return finder


def _metadata(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text())
    compatible = meta.get('measurement_digest') == _measurement_digest()
    registry = json.loads((Path(__file__).parent / 'workspace_compatibility.json').read_text())
    legacy = registry['implementations']
    # A change to acceptance need not invalidate the unchanged prepared mesh.
    # Only explicitly audited preparation-compatible measurements are accepted.
    compatible |= registry.get('measurements', {}).get(meta.get('measurement_digest')) == _measurement_digest()
    compatible |= legacy.get(meta.get('implementation_digest')) == _measurement_digest()
    if meta['schema'] != SCHEMA or not (compatible or meta.get('implementation_digest') == _digest()):
        raise ValueError('Prepared workspace does not match this implementation; prepare a new workspace')
    return meta


def _save_array(directory, name, value):
    """Uncompressed binary NPY preserves direct read-only mmap sharing."""
    np.save(Path(directory) / (name + '.npy'), value, allow_pickle=False)


def _load_array(directory, name):
    directory = Path(directory)
    plain = directory / (name + '.npy')
    if plain.exists():
        return np.load(plain, mmap_mode='r', allow_pickle=False)
    # One decompressed copy per node, shared by that node's MPI ranks. Use a
    # node-local disk, not a private cache for each rank or a RAM-backed tmpfs.
    identity = json.loads((directory / 'manifest.json').read_text())['workspace_id']
    base = Path(os.environ.get('SHOCK_WORKSPACE_CACHE', tempfile.gettempdir()))
    cache = base / ('shockfinder-mmap-' + identity)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (name + '.npy')
    with (cache / (name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            partial = cache / (name + '.partial')
            with gzip.open(directory / (name + '.npy.gz'), 'rb') as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            os.replace(partial, target)
    return np.load(target, mmap_mode='r', allow_pickle=False)


def prepare_workspace(cell, finder, directory, *, task_batches=64, spatial_order=True):
    """Write a new, reusable workspace on a filesystem visible to all workers.

    Call once on one process, preferably in a compute-node PBS preparation job.
    ``directory`` must not exist. Partial workspaces have no manifest and cannot
    be run. Input units, global normalization, leaf IDs, and original numerical
    batches are preserved. No input pickle or RUR object is sent to workers.
    """
    if isinstance(task_batches, bool) or int(task_batches) != task_batches or task_batches < 1:
        raise ValueError('task_batches must be a positive integer')
    finder._validate_settings()
    if finder.boundary != 'open':
        raise ValueError("boundary must be 'open'")
    if _profile_geometry is None:
        raise RuntimeError('Build the native extensions before preparing parallel work')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    finder._progress('ShockFinder: preparing complete shared AMR background')
    arrays = finder._extract_amr_arrays(cell)
    selected = arrays.pop('selected_indices')
    if not len(selected):
        raise ValueError('Parallel preparation requires at least one retained leaf')
    pressure, _ = finder._thermal_pressure_values(cell, selected, arrays)
    context = profile.prepare(arrays, finder, pressure)
    for name in _CONTEXT_ARRAYS:
        _save_array(directory, name, context[name])
    for name in _GEOMETRY_ARRAYS:
        _save_array(directory, 'geometry_' + name, getattr(context['geometry'], name))
    _save_array(directory, 'selected_indices', selected)
    _save_array(directory, 'level', arrays['level'])
    batch = finder.profile_batch_size
    starts = np.arange(0, len(context['seeds']), batch, dtype=np.int64)
    if spatial_order and len(starts):
        # Sort *whole original batches*, never repartition their seed members.
        # Ten-bit Morton keys improve locality without affecting any measurement.
        lo = context['pos'].min(axis=0)
        span = np.maximum(context['pos'].max(axis=0) - lo, context['dx'].min())
        code = np.zeros(len(starts), dtype=np.uint64)
        for first in range(0, len(starts), 262144):
            last = min(first + 262144, len(starts))
            points = context['pos'][context['seeds'][starts[first:last]]]
            quantized = np.minimum(((points - lo) / span * 1023).astype(np.uint64), 1023)
            for bit in range(10):
                for axis in range(3):
                    code[first:last] |= ((quantized[:, axis] >> bit) & 1) << (3 * bit + axis)
        starts = starts[np.argsort(code, kind='stable')]
    _save_array(directory, 'batch_order', starts)
    settings = {name: _json_setting(getattr(finder, name)) for name in vars(ShockFinder())}
    # A workspace is valid only after every array has been written successfully.
    meta = dict(schema=SCHEMA, version=__version__, implementation_digest=_digest(),
                measurement_digest=_measurement_digest(), storage='npy',
                workspace_id=uuid.uuid4().hex, settings=settings,
                cells=len(selected), seeds=len(context['seeds']),
                known_normals=int(context['known'].sum()),
                diagnostics=dict(context['diagnostics']),
                scales={k: context[k] for k in ('density_scale', 'speed_scale', 'qfactor')},
                task_batches=int(task_batches), batches=len(starts),
                tasks=(len(starts) + task_batches - 1) // task_batches,
                spatial_order=bool(spatial_order),
                background='complete_read_only_mmap',
                array_bytes=sum(p.stat().st_size for p in directory.glob('*.npy')),
                preparation_seconds=time.perf_counter() - start)
    (directory / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def load_workspace(directory, *, cache_entries=262144):
    """Attach read-only file-backed shared arrays without rebuilding the index."""
    directory = Path(directory)
    meta = _metadata(directory)
    if _profile_geometry is None:
        raise RuntimeError('Build native profile geometry on this Python environment')
    context = {k: _load_array(directory, k)
               for k in _CONTEXT_ARRAYS}
    geometry = NativeLeafGeometry.__new__(NativeLeafGeometry)
    geometry.pos, geometry.dx = context['pos'], context['dx']
    for name in _GEOMETRY_ARRAYS:
        setattr(geometry, name, _load_array(directory, 'geometry_' + name))
    geometry.kernel = _profile_geometry.profile_geometry_kernel
    geometry.bind_sample_cache(context['values'], cache_entries)
    context.update(meta['scales'], geometry=geometry, diagnostics=meta['diagnostics'])
    starts = _load_array(directory, 'batch_order')
    return context, _finder(meta['settings']), starts, meta


class _Worker:
    def __init__(self, directory, cache_entries):
        self.context, self.finder, self.starts, self.meta = load_workspace(
            directory, cache_entries=cache_entries)
        self.normals = {}

    def run(self, task):
        task = int(task)
        if not 0 <= task < self.meta['tasks']:
            raise ValueError('Invalid seed task')
        geometry = self.context['geometry']
        old_hits, old_queries = geometry.cache_hits, geometry.cache_queries
        start = task * self.meta['task_batches']
        best, counts, _ = profile.scan(self.context, self.finder,
            self.starts[start:start + self.meta['task_batches']], normal_cache=self.normals)
        counts['sample_cache_hits'] = geometry.cache_hits - old_hits
        counts['sample_cache_queries'] = geometry.cache_queries - old_queries
        return task, best, counts


def _initialize_worker(directory, cache_entries):
    global _WORKER
    _WORKER = _Worker(directory, cache_entries)


def _run_task(task):
    return _WORKER.run(task)


class _Collector:
    def __init__(self, directory, show_progress):
        self.directory = Path(directory)
        self.meta = _metadata(directory)
        self.best, self.counts = {}, Counter()
        self.seen = set()
        self.show_progress = show_progress
        self.start = self.last = time.perf_counter()

    def add(self, packet):
        task, best, counts = packet
        if task in self.seen or not 0 <= task < self.meta['tasks']:
            raise RuntimeError('Duplicate or invalid completed seed task')
        self.seen.add(task)
        self.counts.update(counts)
        for cell, fit in best.items():
            old = self.best.get(cell)
            if old is None or profile.fit_precedes(fit, old):
                self.best[cell] = fit
        now = time.perf_counter()
        if self.show_progress and (now - self.last >= 60 or len(self.seen) == self.meta['tasks']):
            done = self.counts['processed_seeds']
            print(f"Distributed profiles: {done:,}/{self.meta['seeds']:,} seeds "
                  f"({done/max(now-self.start, 1e-12):.1f} seeds/s; "
                  f"{len(self.best):,} unique centers)", flush=True)
            self.last = now

    def finish(self, extra_normals, backend, workers):
        if len(self.seen) != self.meta['tasks'] or self.counts['processed_seeds'] != self.meta['seeds']:
            raise RuntimeError('Incomplete seed coverage; refusing to assemble a partial result')
        fits, diagnostics = profile.finish({'diagnostics': self.meta['diagnostics']},
            self.best, self.counts, self.meta['known_normals'] + len(extra_normals))
        diagnostics.update(parallel_workers=workers, parallel_backend=backend,
                           complete_background=True, seed_tasks=len(self.seen))
        arrays = {k: _load_array(self.directory, k)
                  for k in ('pos', 'dx', 'level', 'selected_indices')}
        finder = _finder(self.meta['settings'])
        result = finder._assemble_result(arrays, arrays['selected_indices'], fits, diagnostics)
        result.parallel_timings = dict(scan_and_assembly=time.perf_counter() - self.start,
                                       preparation=self.meta['preparation_seconds'])
        return result


def find_parallel(directory, *, workers=1, cache_entries=262144, show_progress=True):
    """Scan a prepared workspace on one node using spawned shared-mesh workers.

    Set OMP_NUM_THREADS before launching Python. ``workers * threads`` must fit
    the allocation. A new process is spawned; initialized OpenMP is never forked.
    """
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError('workers must be a positive integer')
    collector = _Collector(directory, show_progress)
    extra = set()
    if workers == 1:
        worker = _Worker(directory, cache_entries)
        for task in range(collector.meta['tasks']):
            collector.add(worker.run(task))
        extra.update(worker.normals)
    else:
        # Return lazy normal IDs with each packet; no mutable global normal array.
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_worker, initargs=(str(directory), cache_entries)) as pool:
            tasks = iter(range(collector.meta['tasks']))
            pending = {pool.submit(_run_task_with_normals, t)
                       for t in list(_take(tasks, 2 * workers))}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    packet, normal_ids = future.result()
                    collector.add(packet)
                    extra.update(normal_ids)
                    for task in _take(tasks, 1):
                        pending.add(pool.submit(_run_task_with_normals, task))
    return collector.finish(extra, 'spawn_mmap', int(workers))


def _take(iterator, n):
    for _ in range(n):
        try:
            yield next(iterator)
        except StopIteration:
            return


def _run_task_with_normals(task):
    before = set(_WORKER.normals)
    packet = _WORKER.run(task)
    return packet, list(set(_WORKER.normals) - before)


def find_mpi(directory, *, comm=None, cache_entries=262144, show_progress=True):
    """Collective MPI scan; return a ShockResult on rank zero, None elsewhere.

    Rank zero schedules and merges sparse packets. All other ranks compute.
    Every rank must call this function; the workspace must be shared and immutable.
    No fixed halo or rank boundary can truncate a profile or a final-center move.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD if comm is None else comm
    rank, size = comm.Get_rank(), comm.Get_size()
    if size == 1:
        return find_parallel(directory, cache_entries=cache_entries, show_progress=show_progress)
    error = None
    try:
        meta = _metadata(directory)
        worker = _Worker(directory, cache_entries) if rank else None
        identity = (meta['workspace_id'], meta['implementation_digest'], _measurement_digest())
    except Exception:
        error = traceback.format_exc()
        identity = None
    checks = comm.allgather((identity, error))
    errors = [e for _, e in checks if e]
    if errors or len({i for i, _ in checks}) != 1:
        raise RuntimeError('MPI workspace preflight failed: ' + '\n'.join(errors or ['workspace mismatch']))
    tag_work, tag_done = 710, 711
    global_error = None
    if rank == 0:
        collector = _Collector(directory, show_progress)
        tasks = iter(range(meta['tasks']))
        active = set()
        extra = set()
        for other in range(1, size):
            task = next(tasks, None)
            comm.send(task, dest=other, tag=tag_work)
            if task is not None:
                active.add(other)
        while active:
            status = MPI.Status()
            packet, normal_ids, error = comm.recv(source=MPI.ANY_SOURCE, tag=tag_done, status=status)
            other = status.Get_source()
            if error:
                global_error = global_error or error
            elif global_error is None:
                try:
                    collector.add(packet)
                    extra.update(normal_ids)
                except Exception:
                    global_error = traceback.format_exc()
            task = next(tasks, None) if global_error is None else None
            comm.send(task, dest=other, tag=tag_work)
            if task is None:
                active.remove(other)
        if global_error is None:
            try:
                result = collector.finish(extra, 'mpi_mmap', size - 1)
            except Exception:
                global_error = traceback.format_exc()
    else:
        result = None
        while True:
            task = comm.recv(source=0, tag=tag_work)
            if task is None:
                break
            try:
                before = set(worker.normals)
                packet = worker.run(task)
                comm.send((packet, list(set(worker.normals) - before), None), dest=0, tag=tag_done)
            except Exception:
                comm.send((None, None, traceback.format_exc()), dest=0, tag=tag_done)
    global_error = comm.bcast(global_error, root=0)
    if global_error:
        raise RuntimeError('Distributed profile scan failed:\n' + global_error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--loader', required=True, help='Importable module:function returning (cell, finder)')
    prep.add_argument('--workspace', required=True)
    prep.add_argument('--task-batches', type=int, default=64)
    run = sub.add_parser('run')
    run.add_argument('--workspace', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--mpi', action='store_true')
    run.add_argument('--workers', type=int, default=1)
    run.add_argument('--cache-entries', type=int, default=262144)
    run.add_argument('--analysis-options', help='JSON keyword options for finder.analyze_result')
    args = parser.parse_args()
    if args.command == 'prepare':
        module, function = args.loader.split(':', 1)
        cell, finder = getattr(importlib.import_module(module), function)()
        meta = prepare_workspace(cell, finder, args.workspace, task_batches=args.task_batches)
        print(json.dumps(meta, indent=2), flush=True)
        return
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        expected = os.environ.get('SHOCKFINDER_EXPECT_RANKS')
        if expected and comm.Get_size() != int(expected):
            raise RuntimeError(f'MPI communicator size {comm.Get_size()} != launcher size {expected}; '
                               f'mpi4py extension={MPI.__file__}; runtime={MPI.Get_library_version()}')
        flavor = os.environ.get('SHOCKFINDER_EXPECT_MPI', 'auto')
        runtime = MPI.Get_library_version().lower()
        if flavor != 'auto' and not ((flavor == 'openmpi' and 'open mpi' in runtime) or
                                     (flavor == 'mpich' and 'mpich' in runtime)):
            raise RuntimeError('Loaded MPI runtime does not match the configured launcher: ' + runtime)
    rank = comm.Get_rank() if comm is not None else 0
    # Fail collectively before workers begin if the output would overwrite data.
    preflight = None
    if rank == 0:
        try:
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            if args.analysis_options:
                json.loads(Path(args.analysis_options).read_text())
        except Exception:
            preflight = traceback.format_exc()
    if comm is not None:
        preflight = comm.bcast(preflight, root=0)
    if preflight:
        raise RuntimeError(preflight)
    start = time.perf_counter()
    result = (find_mpi(args.workspace, comm=comm, cache_entries=args.cache_entries) if args.mpi else
              find_parallel(args.workspace, workers=args.workers, cache_entries=args.cache_entries))
    if rank == 0:
        product = result
        diagnostics = dict(result.diagnostics)
        if args.analysis_options:
            settings = _metadata(args.workspace)['settings']
            product = _finder(settings).analyze_result(result,
                timings={'detection_total': time.perf_counter() - start},
                **json.loads(Path(args.analysis_options).read_text()))
        # Protocol 5 keeps dense NumPy arrays efficient; only rank zero writes.
        with Path(args.output).open('xb') as stream:
            pickle.dump(product, stream, protocol=5)
        summary = dict(host=socket.gethostname(), ranks=comm.Get_size() if comm else args.workers,
                       elapsed_seconds=time.perf_counter() - start,
                       diagnostics=diagnostics, output=str(Path(args.output).resolve()))
        Path(str(args.output) + '.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
