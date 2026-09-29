#!/usr/bin/env bash
# Build every Fortran extension into shocktest/ using the active Python.
# Usage: ./f2py.sh   (or PYTHON=/path/to/python ./f2py.sh)
# Requires NumPy/f2py, GNU Fortran, and libgomp.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${project_dir}/shocktest"

python_bin="${PYTHON:-python}"
for source in fortran/*.f90; do
    module_name="_${source##*/}"
    module_name="${module_name%.f90}"
    flags="-O3 -fopenmp"
    if [[ "$module_name" == "_merger_neighbors" ]]; then
        # Preserve rounding at the exact AMR contact/normal thresholds.
        flags+=" -ffp-contract=off"
    fi
    printf 'Building %s from %s\n' "$module_name" "$source"
    "$python_bin" -m numpy.f2py -c "$source" -m "$module_name" \
        --f90flags="$flags" -lgomp
done
