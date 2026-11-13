"""Launch a configured ShockFinder build: OpenMP, MPI, or hybrid execution."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .build_config import active_manifest, runtime_environment, MODES


def execution_plan(path, manifest, args):
    config = manifest['config']
    env = runtime_environment(config)
    threads = args.threads
    if threads < 1 or args.ranks < 1 or args.workers < 1:
        raise ValueError('threads, ranks, and workers must be positive')
    if args.mode == 'mpi' and threads != 1:
        raise ValueError('MPI-only execution requires --threads 1')
    if args.mode == 'omp' and args.ranks != 1:
        raise ValueError('OpenMP-only execution requires --ranks 1')
    if args.mode != 'omp' and args.workers != 1:
        raise ValueError('MPI ranks are the workers; omit --workers')
    if threads > 1 and not manifest['openmp']:
        raise ValueError('This build has no OpenMP; rebuild with --mode omp or hybrid')
    if args.mode != 'omp' and not manifest['mpi']:
        raise ValueError('This build has no configured MPI; rebuild with --mode mpi or hybrid')
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               SHOCKFINDER_BUILD_MANIFEST=str(path))
    env.setdefault('OMP_PROC_BIND', 'spread')
    env.setdefault('OMP_PLACES', 'cores')
    # Ensure spawned processes and remote ranks import the selected checkout.
    root = str(Path(__file__).resolve().parents[1])
    paths = config['mpi']['python_paths'] + [root] + env.get('PYTHONPATH', '').split(os.pathsep)
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    command = [manifest['python'], '-m', 'shocktest.distributed', 'run',
               '--workspace', str(Path(args.workspace).resolve()),
               '--output', str(Path(args.output).resolve()),
               '--cache-entries', str(args.cache_entries)]
    if args.analysis_options:
        command += ['--analysis-options', str(Path(args.analysis_options).resolve())]
    if args.mode == 'omp':
        command += ['--workers', str(args.workers)]
    else:
        env.update(SHOCKFINDER_EXPECT_RANKS=str(args.ranks),
                   SHOCKFINDER_EXPECT_MPI=config['mpi']['flavor'])
        launcher_args = [a.replace('{threads}', str(threads)).replace('{ranks}', str(args.ranks))
                         for a in config['mpi']['launcher_args']]
        if not launcher_args and config['mpi']['flavor'] == 'openmpi':
            launcher_args = ['--map-by', f'slot:PE={threads}', '--bind-to', 'core']
        launcher = [config['mpi']['launcher'], *launcher_args, '-n', str(args.ranks)]
        if config['mpi']['flavor'] == 'openmpi':
            # PBS starts remote ranks in a fresh shell; export the selected
            # Python/MPI paths explicitly rather than relying on activation.
            names = ('PYTHONPATH', 'PYTHONNOUSERSITE', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',
                     'MPI4PY_MPIABI', 'MPI4PY_LIBMPI',
                     'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES',
                     'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'SHOCKFINDER_BUILD_MANIFEST',
                     'SHOCKFINDER_EXPECT_RANKS', 'SHOCKFINDER_EXPECT_MPI')
            for name in dict.fromkeys((*names, *config.get('environment', {}))):
                if name in env:
                    launcher += ['-x', name]
        command = launcher + command + ['--mpi']
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run')
    run.add_argument('--mode', choices=MODES, required=True)
    run.add_argument('--threads', type=int, default=1)
    run.add_argument('--ranks', type=int, default=1)
    run.add_argument('--workers', type=int, default=1)
    run.add_argument('--workspace', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--analysis-options')
    run.add_argument('--cache-entries', type=int, default=262144)
    run.add_argument('--dry-run', action='store_true')
    check = sub.add_parser('check')
    check.add_argument('--mpi', action='store_true', help='Initialize MPI; run within an allocation')
    args = parser.parse_args()
    path, manifest = active_manifest()
    if args.action == 'check':
        print(json.dumps(manifest, indent=2), flush=True)
        env = runtime_environment(manifest['config'])
        code = ('import sys, shocktest; from shocktest import _profile_geometry; '
                'print("Python:", sys.executable); print("Package:", shocktest.__file__); '
                'assert _profile_geometry is not None; '
                'assert hasattr(_profile_geometry.profile_geometry_kernel,"sample_batch_reuse"); '
                'assert hasattr(_profile_geometry.profile_model_kernel,"screen_batch"); '
                'assert hasattr(_profile_geometry.profile_model_kernel,"pair_support_batch"); '
                'assert hasattr(_profile_geometry.profile_model_kernel,"upstream_policy_version"), '
                '"Rebuild the v0.6.3 native profile extension"; '
                'assert _profile_geometry.profile_model_kernel.upstream_policy_version()==3; '
                'print("Native kernels OK")')
        if args.mpi:
            if not manifest['mpi']:
                raise ValueError('Build with --mode mpi or hybrid first')
            code += ('; from mpi4py import MPI; text=MPI.Get_library_version(); '
                     'print("MPI:", text); '
                     f'flavor={manifest["config"]["mpi"]["flavor"]!r}; '
                     'assert flavor=="auto" or (flavor=="openmpi" and "Open MPI" in text) '
                     'or (flavor=="mpich" and "MPICH" in text)')
        subprocess.run([manifest['python'], '-c', code], env=env, check=True)
        return
    command, env = execution_plan(path, manifest, args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print(json.dumps({k: env[k] for k in ('OMP_NUM_THREADS', 'PYTHONPATH', 'LD_LIBRARY_PATH') if k in env}, indent=2))
    else:
        raise SystemExit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()
