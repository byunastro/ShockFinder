#!/usr/bin/env bash
# Build every Fortran extension for one Python/NumPy environment.
# Usage: PYTHON=/path/to/python ./f2py.sh (defaults to python3).
# Defaults require NumPy/f2py, GNU Fortran, and libgomp.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
exec "$python_bin" "${project_dir}/tools/build_extensions.py"
