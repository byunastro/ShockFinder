"""Portable build configuration and recorded execution environment.

This module uses only the standard library so the build driver can load it
before importing NumPy or any compiled ShockFinder extension.
"""
import json
import os
from pathlib import Path
import shutil
import sys

MODES = ('omp', 'mpi', 'hybrid')


def expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def read_config(path=None, mode=None):
    supplied = expand(json.loads(Path(path).read_text())) if path else {}
    allowed = {'schema', 'python', 'mode', 'cc', 'fc', 'backend', 'fortran_flags',
               'precision_flags', 'free_line_length_flag', 'openmp', 'mpi', 'environment'}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError('Unknown build settings: ' + ', '.join(sorted(unknown)))
    if supplied.get('schema', 1) != 1:
        raise ValueError('Unsupported build configuration schema')
    config = dict(schema=1, python=sys.executable, mode='omp',
                  cc=os.environ.get('CC', 'cc'), fc=os.environ.get('FC', 'gfortran'),
                  backend='meson', fortran_flags=['-O3'],
                  precision_flags=['-ffp-contract=off'], free_line_length_flag='-ffree-line-length-none',
                  openmp={'flags': ['-fopenmp'], 'libraries': ['gomp']},
                  mpi={'flavor': 'auto', 'mpicc': 'mpicc', 'launcher': 'mpirun',
                       'launcher_args': [], 'python_paths': [], 'library_dirs': [],
                       'install': 'none', 'requirement': 'mpi4py>=4.0'}, environment={})
    for key, value in supplied.items():
        if key in ('openmp', 'mpi'):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f'Unknown or malformed {key} settings')
            config[key].update(value)
        else:
            config[key] = value
    config['python'] = config['python'] or sys.executable
    if mode:
        config['mode'] = mode
    if config['mode'] not in MODES:
        raise ValueError('mode must be omp, mpi, or hybrid')
    if config['backend'] not in ('meson', 'distutils'):
        raise ValueError('backend must be meson or distutils')
    if config['mpi']['flavor'] not in ('auto', 'openmpi', 'mpich'):
        raise ValueError('mpi.flavor must be auto, openmpi, or mpich')
    if config['mpi']['install'] not in ('none', 'wheel', 'source'):
        raise ValueError('mpi.install must be none, wheel, or source')
    for owner, key in ((config, 'fortran_flags'), (config, 'precision_flags'),
                       (config['openmp'], 'flags'), (config['openmp'], 'libraries'),
                       (config['mpi'], 'launcher_args'), (config['mpi'], 'python_paths'),
                       (config['mpi'], 'library_dirs')):
        if not isinstance(owner[key], list) or not all(isinstance(x, str) for x in owner[key]):
            raise ValueError(f'{key} must be a list of strings')
    if not isinstance(config['environment'], dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in config['environment'].items()):
        raise ValueError('environment must map names to string values')
    flags = config['fortran_flags'] + config['precision_flags']
    if any(f in flags for f in ('-Ofast', '-ffast-math', '-ffp-contract=fast')):
        raise ValueError('Fast-math flags can change shock selection; use conservative floating-point flags')
    return config


def resolve_program(program, env=None):
    env = os.environ if env is None else env
    found = shutil.which(program, path=env.get('PATH'))
    if found is None:
        raise ValueError(f'Executable not found: {program}')
    # Do not resolve symlinks: MPI wrappers may depend on their invocation name.
    return os.path.abspath(found)


def runtime_environment(config, base=None):
    env = dict(os.environ if base is None else base)
    env.update(config.get('environment', {}))
    for key, paths in (('PYTHONPATH', config['mpi']['python_paths']),
                       ('LD_LIBRARY_PATH', config['mpi']['library_dirs'])):
        if paths:
            env[key] = os.pathsep.join(paths + ([env[key]] if env.get(key) else []))
    return env


def module_flags(config, source):
    flags = list(config['fortran_flags'])
    if config['mode'] in ('omp', 'hybrid'):
        flags += config['openmp']['flags']
    if source in ('profile_geometry', 'merger_neighbors'):
        flags += config['precision_flags']
    if source == 'profile_geometry' and config['free_line_length_flag']:
        flags.append(config['free_line_length_flag'])
    return flags


def active_manifest():
    from ._native import extension_directory
    path = Path(os.environ.get('SHOCKFINDER_BUILD_MANIFEST',
                              extension_directory() / 'build_manifest.json'))
    if not path.is_file():
        raise ValueError('No configured build found for this Python/NumPy environment. Run build.py first.')
    manifest = json.loads(path.read_text())
    if Path(manifest['python']).absolute() != Path(sys.executable).absolute():
        raise ValueError(f"Use the build interpreter: {manifest['python']}")
    return path, manifest
