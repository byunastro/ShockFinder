"""Build and verify the ShockFinder F2PY modules for this interpreter."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "shocktest"
SOURCES = ("shockfinder", "merger_neighbors", "profile_geometry")


def fortran_build_environment(flags: str) -> dict[str, str]:
    """Pass Fortran flags through Meson as well as F2PY's CLI option."""
    env = os.environ.copy()
    # NumPy 1.26's Meson template ignores --f90flags. Meson reads FFLAGS
    # during its fresh setup, so keep any existing flags and append ours.
    env["FFLAGS"] = " ".join(
        part for part in (env.get("FFLAGS", ""), flags.strip()) if part
    )
    return env


def install_extensions(built: list[tuple[Path, Path]], destination: Path) -> None:
    """Stage on the destination filesystem before replacing existing modules."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".f2py-install-", dir=destination
    ) as staging:
        staged = []
        for artifact, target in built:
            staged_artifact = Path(staging) / artifact.name
            shutil.copy2(artifact, staged_artifact)
            staged.append((staged_artifact, target))
        for staged_artifact, target in staged:
            os.replace(staged_artifact, target)


def main(argv=None) -> None:
    import argparse
    import json
    import shlex
    import uuid
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--mode', choices=('omp', 'mpi', 'hybrid'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--modules', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args(argv)
    config_path = args.config or (ROOT / 'build_config.json' if (ROOT / 'build_config.json').exists() else None)
    spec = importlib.util.spec_from_file_location('shockfinder_build_config', PACKAGE / 'build_config.py')
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    config = settings.read_config(config_path, args.mode)
    env = settings.runtime_environment(config)
    python = settings.resolve_program(config['python'], env)
    # Module files can put /bin/python ahead of the active conda interpreter.
    # Re-execute before importing NumPy or computing environment-specific paths.
    if Path(python).absolute() != Path(sys.executable).absolute():
        command = [python, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        os.execve(python, command, env)
    config['python'] = sys.executable
    if sys.version_info < (3, 10):
        raise SystemExit('Select Python >=3.10 in the build configuration')
    for key in ('cc', 'fc'):
        config[key] = settings.resolve_program(config[key], env)
    versions = {}
    for key in ('cc', 'fc'):
        probe = subprocess.run([config[key], '--version'], env=env, check=True,
                               text=True, capture_output=True)
        versions[key] = probe.stdout.splitlines()[0]
    mpi_enabled = config['mode'] in ('mpi', 'hybrid')
    if mpi_enabled:
        config['mpi']['launcher'] = settings.resolve_program(config['mpi']['launcher'], env)
        if config['mpi']['install'] == 'source':
            config['mpi']['mpicc'] = settings.resolve_program(config['mpi']['mpicc'], env)
        else:
            found = shutil.which(config['mpi']['mpicc'], path=env.get('PATH'))
            if found:
                config['mpi']['mpicc'] = os.path.abspath(found)
        version = subprocess.run([config['mpi']['launcher'], '--version'], env=env,
                                 check=True, capture_output=True, text=True)
        versions['mpi_launcher'] = version.stdout + version.stderr
        if config['mpi']['flavor'] == 'auto':
            text = versions['mpi_launcher'].lower()
            if 'open mpi' in text or 'openrte' in text:
                config['mpi']['flavor'] = 'openmpi'
            elif 'hydra' in text or 'mpich' in text:
                config['mpi']['flavor'] = 'mpich'
            else:
                raise ValueError('Set mpi.flavor explicitly for this MPI launcher')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f'Install dependencies with {sys.executable} -m pip install -e ".[native]"') from exc
    if config['backend'] == 'distutils' and (sys.version_info >= (3, 12)
            or np.lib.NumpyVersion(np.__version__) >= '2.5.0'):
        raise ValueError('Select backend=meson for Python >=3.12 or NumPy >=2.5')
    spec = importlib.util.spec_from_file_location('shocktest_native_paths', PACKAGE / '_native.py')
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    destination = paths.extension_directory()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    # A clean Meson directory makes compiler/flag changes take effect.
    work = ROOT / 'build' / 'native' / uuid.uuid4().hex
    plans = []
    for source in args.modules:
        flags = settings.module_flags(config, source)
        if config_path is None and 'SHOCKFINDER_F90FLAGS' in os.environ:
            flags = shlex.split(os.environ['SHOCKFINDER_F90FLAGS'])
            if config['mode'] == 'mpi' and any('openmp' in f for f in flags):
                raise ValueError('Remove OpenMP from SHOCKFINDER_F90FLAGS for an MPI-only build')
            if source in ('profile_geometry', 'merger_neighbors'):
                flags += config['precision_flags']
            if source == 'profile_geometry':
                flags.append(config['free_line_length_flag'])
        name = '_' + source
        command = [sys.executable, '-m', 'numpy.f2py', '-c',
                   str(PACKAGE / 'fortran' / (source + '.f90')), '-m', name,
                   '--backend', config['backend'], '--f90flags=' + shlex.join(flags)]
        if source == 'profile_geometry':
            command += ['only:', 'build_index', 'locate_batch', 'sample_batch',
                        'sample_batch_cached', 'sample_batch_reuse', 'derivatives',
                        'derivatives_at', 'fit_windows', 'mass_root', 'screen_batch', ':']
        libraries = config['openmp']['libraries'] if config['mode'] != 'mpi' else []
        if config_path is None and 'SHOCKFINDER_OPENMP_LIB' in os.environ and config['mode'] != 'mpi':
            libraries = [os.environ['SHOCKFINDER_OPENMP_LIB']] if os.environ['SHOCKFINDER_OPENMP_LIB'] else []
        command += ['-l' + lib for lib in libraries]
        build_env = dict(env, CC=config['cc'], FC=config['fc'], F77=config['fc'], F90=config['fc'],
                         FFLAGS=shlex.join(flags))
        plans.append((source, command, build_env))
    manifest = dict(schema=1, python=sys.executable, numpy=np.__version__,
                    config=config, compiler_versions=versions, modules=args.modules,
                    openmp=config['mode'] != 'mpi', mpi=mpi_enabled,
                    extension_directory=str(destination),
                    commands=[command for _, command, _ in plans])
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    work.mkdir(parents=True)
    if mpi_enabled and config['mpi']['install'] != 'none':
        # Isolate MPI bindings from preinstalled MPICH/Open MPI packages. Never
        # replace the cluster's MPI runtime or the user's other conda packages.
        target = destination / 'mpi-python' / uuid.uuid4().hex
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                   '--target', str(target), config['mpi']['requirement']]
        command += (['--only-binary=:all:'] if config['mpi']['install'] == 'wheel'
                    else ['--no-binary=mpi4py'])
        install_env = dict(env, MPICC=config['mpi']['mpicc'],
                           MPI4PY_BUILD_MPICC=config['mpi']['mpicc'])
        subprocess.run(command, check=True, env=install_env)
        config['mpi']['python_paths'].insert(0, str(target))
        env = settings.runtime_environment(config)
    if mpi_enabled:
        # Inspect package availability without importing MPI or calling MPI_Init.
        subprocess.run([sys.executable, '-c',
            'import mpi4py; print("MPI Python package:", mpi4py.__version__, mpi4py.__file__)'],
            check=True, env=env)
    print(json.dumps(manifest, indent=2), flush=True)
    built = []
    for source, command, build_env in plans:
        print('Building', source, flush=True)
        with (work / (source + '.log')).open('w') as log:
            try:
                subprocess.run(command, cwd=work, check=True, env=build_env,
                               stdout=log, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                raise SystemExit(f'Build failed. See {work / (source + ".log")}')
        artifact = work / ('_' + source + suffix)
        if not artifact.is_file():
            raise RuntimeError(f'F2PY did not produce {artifact}')
        built.append((artifact, destination / artifact.name))
    install_extensions(built, destination)
    verify = ('import sys, pathlib, shocktest; '
              'root=pathlib.Path(sys.argv[1]).resolve(); '
              'assert pathlib.Path(shocktest.__file__).resolve().parent == root / "shocktest"; '
              'modules=[getattr(shocktest, "_"+name) for name in sys.argv[2:]]; '
              'assert all(m is not None for m in modules); '
              'print("Import verified:", *[m.__file__ for m in modules])')
    subprocess.run([sys.executable, '-c', verify, str(ROOT), *args.modules],
                   cwd=ROOT, check=True, env=env)
    # Store the resolved environment only after successful compilation/import.
    # This file is per Python/NumPy environment, beside its native extensions.
    temporary = destination / 'build_manifest.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, destination / 'build_manifest.json')
    print('Build manifest:', destination / 'build_manifest.json', flush=True)
    print('MPI runtime validation is deferred to the allocated job (launch check --mpi).')


if __name__ == '__main__':
    main()
