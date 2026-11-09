# Configuring and running ShockFinder

The public build entry is `build.py`. It uses NumPy F2PY and Meson to compile
GNU Fortran kernels. MPI task distribution is implemented by mpi4py, so the
Fortran compiler and MPI C wrapper are selected separately. GNU Fortran on
Linux/macOS is the tested toolchain; other Fortran compilers need compatible
flags and have not been validated.

## Start from a checkout

```bash
git clone https://github.com/byunastro/ShockFinder.git
cd ShockFinder
RUN_PYTHON=/absolute/path/to/environment/bin/python
"$RUN_PYTHON" -m pip install -e '.[native]'
cp build_config.example.json build_config.json
```

Edit the local JSON file before building. `build_config.json` is ignored by
Git. Use absolute executable and library paths when module loading changes
`PATH`. JSON strings can contain environment variables such as
`"$CONDA_PREFIX/bin/python"`; load modules/activate the environment before
building. The driver re-executes the configured Python **before** importing
NumPy, which prevents a system `python` from building against a different
Conda environment. Runtime commands should use this same interpreter.

```bash
"$RUN_PYTHON" build.py --config build_config.json --mode hybrid --dry-run
"$RUN_PYTHON" build.py --config build_config.json --mode hybrid
"$RUN_PYTHON" -m shocktest.launch check
```

No MPI job is launched during compilation. The build checks executables and
package availability without importing `mpi4py.MPI` or calling `MPI_Init`.
Perform compilation on a host permitted by your site's policy. Runtime checks
and calculations go inside a scheduler allocation.

## Configuration fields

| Field | Meaning |
|---|---|
| `python` | Exact interpreter; empty selects the interpreter invoking the driver |
| `cc`, `fc` | C and Fortran executable paths, or names resolved from configured `PATH` |
| `mode` | `omp`, `mpi`, or `hybrid`; overridden by `--mode` |
| `backend` | `meson` recommended; `distutils` only for old supported Python/NumPy combinations |
| `fortran_flags` | Common flags, default `-O3` |
| `precision_flags` | Profile/neighbor rounding controls, default `-ffp-contract=off` |
| `free_line_length_flag` | GNU free-source line-length flag for the profile kernel |
| `openmp.flags`, `openmp.libraries` | OpenMP compile flags and library names; omitted in MPI-only builds |
| `mpi.flavor` | `auto`, `openmpi`, or `mpich`; auto detects the launcher version |
| `mpi.mpicc` | MPI C wrapper, used when building mpi4py from source |
| `mpi.launcher`, `mpi.launcher_args` | MPI executable and argument list |
| `mpi.python_paths` | mpi4py/site directories prepended to the worker's Python path |
| `mpi.library_dirs` | Directories prepended to `LD_LIBRARY_PATH` |
| `mpi.install` | `none` (existing package), `wheel`, or `source` |
| `mpi.requirement` | pip requirement used for optional installation |
| `environment` | Additional environment variables, e.g. `PYTHONNOUSERSITE`, `LDFLAGS`, compiler runtime paths |

Fast-math flags `-Ofast`, `-ffast-math`, and `-ffp-contract=fast` are rejected
because they can change physical thresholds. Flags and compiler versions are
recorded. Builds use clean Meson directories so changes to compilers or flags
take effect. Logs remain under `build/native/<id>/`.

`mpi.install=none` is appropriate when an existing mpi4py already matches the
cluster MPI, including a wheel installed in a separate directory listed in
`mpi.python_paths`. With `wheel` or `source`, the selected Python installs only
mpi4py into a fresh private directory beside the native extensions. It does
not replace the cluster's MPI runtime or the environment's existing packages.
The private path is recorded for workers. Source installation sets `MPICC` and
`MPI4PY_BUILD_MPICC`; wheel installation avoids Conda linker/sysroot conflicts.
Neither choice proves runtime compatibility: use the allocated-job MPI check.

## Build and execution modes

| Build mode | Native OpenMP | Configured MPI | Runtime choices |
|---|---|---|---|
| `omp` | Enabled | Not required | `omp` |
| `mpi` | Disabled, no OpenMP libraries linked | Enabled | `mpi`, or single-thread `omp` |
| `hybrid` | Enabled | Enabled | `omp`, `mpi`, `hybrid` |

A hybrid build is recommended when switching execution modes on one machine.
MPI-only execution of that build sets one OpenMP thread per rank. For a binary
that has no OpenMP dependency, compile with `--mode mpi`. The estimator,
profile sampling, and physical acceptance conditions are identical.

The last build in a Python/NumPy environment is the active one. Rebuilding with
a different mode replaces that environment's native modules and manifest;
restart Python afterward. Use different environments/checkouts if you need
simultaneous independent builds. `--modules profile_geometry` is available for
advanced partial builds; use a full build when changing execution modes.

A successful build records the resolved configuration in
`shocktest/_f2py_builds/<environment-id>/build_manifest.json`. The launcher uses
that Python, MPI executable and environment rather than resolving a new
`python` or `mpirun` on each node. The active manifest is specific to the
Python prefix, NumPy installation and platform. Rebuild on the target server;
do not copy macOS binaries to Linux or assume heterogeneous CPUs share an ABI.

## Run a prepared workspace

Use separate scheduler allocations matching each example's resource counts.
These commands perform the scan; they do not allocate CPUs themselves.

```bash
# One process using 48 OpenMP threads, without MPI.
"$RUN_PYTHON" -m shocktest.launch run --mode omp --threads 48 \
    --workspace /shared/prepared --output result_omp.pkl

# Two nodes, six MPI ranks per node and eight threads per rank.
"$RUN_PYTHON" -m shocktest.launch run --mode hybrid --ranks 12 --threads 8 \
    --workspace /shared/prepared --output result_hybrid.pkl

# Two nodes, 48 MPI ranks per node and one thread per rank.
"$RUN_PYTHON" -m shocktest.launch run --mode mpi --ranks 96 --threads 1 \
    --workspace /shared/prepared --output result_mpi.pkl
```

Rank zero coordinates and assembles results; the other ranks scan seeds. A
one-rank MPI job computes locally. OpenMP mode can also use `--workers N` for
spawned shared-mesh processes without MPI; allocate `N * threads` cores.
BLAS threads are set to one to avoid nested oversubscription. `--dry-run`
prints the launch command without initializing MPI. `--analysis-options
options.json` retains the existing post-processing interface.

For Open MPI, an empty `launcher_args` uses `--map-by slot:PE=<threads>
--bind-to core` so each rank receives enough cores for its OpenMP threads.
If your cluster needs different placement, supply the complete site-specific
arguments. `{threads}` and `{ranks}` placeholders expand at runtime. MPICH
uses its launcher's defaults unless you configure placement arguments.

Open MPI workers receive the selected Python/MPI paths and thread controls
via explicit `-x` exports. MPICH Hydra inherits the launcher environment.
Configure shared paths that are available on every allocated node. The worker
checks communicator size and MPI implementation against the selected launcher
before starting a scan. A mismatched MPICH/Open MPI installation fails instead
of independently running duplicate single-rank scans.

## Scheduler use

For PBS hybrid execution, a typical resource request is
`select=2:ncpus=48:mpiprocs=6:ompthreads=8:mem=200gb`. Put the corresponding
hybrid launch command in the job script, after loading the site modules.
For MPI-only, request `mpiprocs=48:ompthreads=1` per 48-core node instead.
Queue, account, walltime and module names are site settings. There is no need
to access the compute node manually: `qsub` allocates nodes and runs the script
there. Open MPI can use PBS's native allocation information.

Within an allocated job, check the actual runtime with:

```bash
"$RUN_PYTHON" -m shocktest.launch check --mpi
```

This initializes MPI in a child process and verifies the loaded library.
A multi-node scientific smoke test or small real workspace should precede a
large production scan. Compilation success alone cannot validate the cluster
network, MPI runtime or scheduler placement.

## Existing workspace data

`prepare_workspace` writes uncompressed binary `.npy` arrays, preserving direct
read-only mmap sharing. It does not make a decompression cache. A shared
filesystem must expose the workspace at the same path on every node. Every
worker retains access to the complete retained background, including across
seed-task boundaries.

Already prepared v0.5.0 NPY workspaces from the distributed release can be used
without rewriting arrays or editing their manifests. Known release fingerprints
are mapped to the same measurement implementation. New workspaces record a
separate measurement fingerprint, so changes to the launcher/build tooling do
not force expensive preparation again. Changes to the scientific preparation,
profile model or geometry still reject an incompatible workspace. Unknown old
fingerprints are not silently accepted. The legacy compressed reader remains
available for old compressed workspaces; new preparations always write NPY.

## References

- [NumPy compiler/environment configuration](https://numpy.org/doc/stable/building/compilers_and_options.html)
- [NumPy F2PY build systems](https://numpy.org/doc/stable/f2py/buildtools/index.html)
- [mpi4py installation and external MPI](https://mpi4py.readthedocs.io/en/stable/install.html)
