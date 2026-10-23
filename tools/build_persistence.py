"""Build the optional C++ flux-persistence helper, separately from f2py."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    if sys.platform not in ('darwin', 'linux'):
        raise SystemExit('The experimental helper currently supports macOS and Linux.')
    root = Path(__file__).resolve().parents[1]
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    target = root / 'shocktest' / ('_flux_persistence' + suffix)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    # Replace only after a successful build; never leave a partial library.
    with tempfile.TemporaryDirectory(prefix='persistence-build-', dir=target.parent) as work:
        output = Path(work) / target.name
        subprocess.run(compiler + ['-O3', '-std=c++11', '-fPIC', shared,
                                  str(root / 'native' / 'flux_persistence.cpp'),
                                  '-o', str(output)], check=True)
        os.replace(output, target)
    print(f'Built {target}')


if __name__ == '__main__':
    main()
