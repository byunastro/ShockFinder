# ShockFinder

ShockFinder is a Python framework for detecting shocks on AMR leaf meshes,
measuring Mach numbers and thermal dissipation, grouping shock structures,
and supporting galaxy matching and time-resolved exposure measurements.
The current detector uses **local profile reconstruction only**: it reconstructs
a subcell shock surface from column mass and adopts a pressure-jump Mach number.
There is no selector for the earlier center-search methods.

## 1. Install and configure

Requirements: Python >= 3.10, NumPy >= 2.0, and SciPy >= 1.13.
For native execution, GNU Fortran, a C compiler, Meson and Ninja are required.
Use the exact Python executable of your intended environment, especially when
cluster modules change `PATH`.

```bash
git clone https://github.com/byunastro/ShockFinder.git
cd ShockFinder
# Replace this with the absolute path of your Python environment.
RUN_PYTHON=/path/to/environment/bin/python
"$RUN_PYTHON" -m pip install -e '.[native]'
cp build_config.example.json build_config.json
```

Edit `build_config.json` to select `python`, `cc`, `fc`, and (for MPI) the MPI
launcher, `mpicc`, library directories and mpi4py paths. The per-machine file
is ignored by Git. The generic example defaults to an OpenMP build without MPI.

```bash
"$RUN_PYTHON" build.py --config build_config.json --mode omp
"$RUN_PYTHON" -m shocktest.launch check
```

To enable all execution modes, set the MPI section for your system and build
with `--mode hybrid`. To compile without OpenMP flags or libraries, choose
`--mode mpi`. This uses serial Fortran inside each MPI rank. These choices
configure execution; the shock estimator is the same.

```bash
"$RUN_PYTHON" build.py --config build_config.json --mode hybrid
# Inspect the resolved compilers, paths and commands without compiling:
"$RUN_PYTHON" build.py --config build_config.json --mode hybrid --dry-run
```

Builds record a `build_manifest.json` beside the environment-specific native
extensions. The launcher uses its Python and MPI configuration. Compilation
never starts an MPI job. Run MPI checks and calculations inside your scheduler's
allocation; submit the launch command in a PBS/Slurm script if appropriate.

```bash
# After preparing a shared workspace, choose the allocated resources:
"$RUN_PYTHON" -m shocktest.launch run --mode omp --threads 48 \
    --workspace /shared/prepared --output result_omp.pkl
"$RUN_PYTHON" -m shocktest.launch run --mode hybrid --ranks 12 --threads 8 \
    --workspace /shared/prepared --output result_hybrid.pkl
"$RUN_PYTHON" -m shocktest.launch run --mode mpi --ranks 96 --threads 1 \
    --workspace /shared/prepared --output result_mpi.pkl
```

A hybrid build supports all three runtime modes. An MPI-only build supports
one thread per rank and does not link OpenMP in its native modules. MPI and
hybrid execution use rank zero as coordinator. Rank/thread placement must
match the scheduler allocation. These examples describe separate allocations,
not three jobs that share the same resources. See [BUILDING.md](BUILDING.md) for
the configuration fields, MPI wheel/source choices, CPU binding and PBS usage.

`python tools/build_extensions.py`, `PYTHON=... ./f2py.sh`, and
`python tools/build_profile_geometry.py` remain compatibility entry points;
they now use the same configuration driver. Restart Python after rebuilding.
For pure Python use, `pip install -e .` and `finder.find(cell)` remain available;
shared-workspace parallel execution requires the native geometry kernel.

## 2. Use

The following example explicitly sets
**commonly used finder parameters to their defaults**, with the purpose of each setting
beside the assignment:

```python
import shocktest

finder = shocktest.ShockFinder()

# Geometry: retain only cells within this inclusive AMR-level range.
finder.minlevel = 0                  # Lowest retained AMR level.
finder.maxlevel = 20                 # Highest retained AMR level.

# Thermodynamics and candidate selection.
finder.mean_molecular_weight = 0.59   # EOS and sound-speed mean molecular weight.
finder.min_temperature = None        # Optional seed-cell lower T cut [K]; None disables it.
finder.min_density = None            # Optional seed-cell lower density cut [Msol/kpc3].
finder.max_density = None            # Optional seed-cell upper density cut [Msol/kpc3].
finder.min_mach = 1.3                # Accepted pressure Mach must be strictly greater.
finder.sampling_tolerance = 0.01      # Relative background-profile fitting tolerance.
finder.seed_bounds = None            # Optional (3,2) coordinate bounds for starting seeds.

# Unit keys used to access the input table; use these units for the full pipeline.
finder.position_unit = "km"          # Positions and cell widths.
finder.velocity_unit = "km/s"        # Velocity components.
finder.temperature_unit = "K"        # Gas temperature.
finder.density_unit = "Msol/kpc3"     # Gas mass density.
finder.thermal_pressure_field = None # Infer thermal pressure from rho, T, and mu if absent.
finder.thermal_pressure_unit = "erg/cm3"  # Unit of an explicitly supplied thermal pressure.

# Memory and progress controls.
finder.profile_batch_size = 32       # Limit temporary profile-sampling arrays.
finder.show_progress = False         # Print input and profile-batch progress.

result = finder.find(cell)
print("Detected centers:", result.shock.sum())
```

The adopted Mach number comes from reconstructed **thermal pressure** states,
with `result.mach_estimator == 'profile_pressure'`. The finder checks positive
entropy production and the normal velocity jump internally. Density, momentum,
and energy residuals and background sensitivity remain in `result.sampling`.
It does not calculate secondary temperature/density Mach numbers or apply the
old Mach-consistency post-filter. Always select using `result.shock` and state
any additional profile-quality selection explicitly.

Level cuts change the retained mesh, so overly restrictive cuts can remove
needed neighbors. Temperature/density cuts affect only detection seeds: other
retained cells remain available to reconstructed profiles. `seed_bounds` also
restricts only starting seeds, in `position_unit`; it does not crop the mesh.
Use all available leaf levels when possible. Inputs have open boundaries;
periodic wrapping is not performed.

### Profile reconstruction

1. Seed locally compressing cells with nonzero temperature gradient and
   `grad(T) · grad(P) > 0`. The normal is `grad(T) / |grad(T)|`, directed from
   upstream to downstream.
2. Sample density, thermal pressure, and momentum conservatively on a common
   virtual AMR mesh. The reference width `h` is the coarsest covering leaf at
   preliminary offsets `0, ±4, ±8, ±12` local cell widths. Sample from `-12h` to
   `+12h` at spacing `h/4`, averaging a central ray and four transverse rays.
   If the transverse stencil is incomplete, use the central ray.
3. Fit resolved upstream/downstream backgrounds in `ln(rho)`, pressure, and
   normal velocity. Locate a subcell surface where the joined density
   backgrounds preserve the measured column mass across the transition.
4. Recompute the normal in the containing leaf and repeat, using two to six
   passes. The accepted profile uses the final leaf's normal. A last-pass
   bounded solution is allowed only if the free mass surface is within `h/4`
   of the current surface; it is explicitly flagged. Retain the best fit per
   containing leaf.
5. Evaluate one-sided states at the reconstructed surface and adopt

   $$\mathcal{M}_P = \left[\frac{(\gamma+1)P_{\rm down}/P_{\rm up}+\gamma-1}{2\gamma}\right]^{1/2}.$$

   Require positive pressure/density jumps, positive entropy production,
   normal velocity-jump agreement within 15% of
   the RH prediction, and `M_P > min_mach`. Density-jump agreement contributes
   to fit ranking rather than imposing a hard rejection inside reconstruction.

An entropy signal smaller than estimated background-model sensitivity is now
retained with `ENTROPY_UNCERTAIN`, rather than silently removed. `result.shock`
includes these tentative measurements. Use and report the quality flags when
selecting a scientific sample. They are diagnostics, not calibrated confidence
levels or ground-truth shock labels. Derivatives are computed for eligible seeds
and lazily for newly located center leaves; sampling always retains all leaves.

Version 0.4.1 reuses identical virtual-cell evaluations within each native
sampling batch. Cache keys include lattice coordinates and the exact sampling
width; the cache is discarded after the batch, so fields cannot become stale.
Restriction, interpolation order, profile fitting, and physical checks remain
the same. Small lookup/derivative batches run serially to avoid OpenMP team
overhead. This is a computational optimization for every shock orientation.
Temporary caching is bounded to 32,768 sample points per call and can use tens
of MiB of additional memory; it does not cache the full simulation.

Rebuild after updating with `python tools/build_profile_geometry.py` and restart
the Python process. `result.diagnostics['geometry_cache_native'] == 1` confirms
the new native entry point is available. Older binaries remain usable through
the original sampling kernel, but do not provide the new cache acceleration.

Version 0.4.2 retains only the best converged model per output leaf during the
scan. It preserves the original quality ranking and tie order while avoiding
storage and export-field calculation for losing duplicate models. The number
of seeded profiles and the physical measurement rules remain unchanged. Each
profile also reuses its trapezoid terms across background pairs; interval sums
retain their original reduction order, without cumulative-sum subtraction.
Scalar density evaluation and background-order classification avoid repeated
NumPy dispatch without changing the model selection rules. Progress
output respects `progress_interval` (seed count), defaults to approximately 5%
increments, and also reports after 60 seconds at the next batch boundary. It
includes processed seeds per second and the current number of unique centers;
it no longer prints and flushes a line for every 32-seed batch.

Version 0.4.3 batches profile preprocessing and runs background fitting, mass
roots, and conservative physical rejection in an OpenMP Fortran batch kernel.
Borderline models and surviving candidates still pass the original Python
physical checks, quality ranking, and diagnostic calculation. Incomplete
profiles and fixed-center retries use the scalar path. No seeds, rays, background
models, or physical acceptance conditions are removed. The batch kernel uses
the same pseudoinverse tables and pairwise integral summation as the reference.
`profile_batch_size` continues to control temporary working memory; the default
remains 32. This is an execution change, not a selectable scientific estimator.

After rebuilding and restarting Python,
`result.diagnostics['model_batch_native'] == 1` indicates that the new native
batch entry point is available. Older binaries automatically retain scalar
model evaluation and do not provide this acceleration. Model-only speedups are
not the same as end-to-end speedups: AMR sampling, input preparation and final
result assembly remain part of a complete detection run.

### Shared-memory and MPI execution (0.5.0)

Large scans can prepare the complete retained background once and distribute
**all original seed batches**. The scientific estimator remains unchanged.
Workers map the same read-only NPY files, so processes on a node share their
physical file pages. Each node must still accommodate the background working
set: this is seed-work distribution, not distributed storage of the AMR mesh.
Every worker can sample across any assigned task boundary, including after
coarse/fine changes or recentering. There is no finite halo cutoff. Physical
input edges and user level cuts retain their existing meaning.

Prepare on one process using the same `cell` and configured `finder` as above:

```python
from shocktest.distributed import prepare_workspace

# Use a new directory on a filesystem visible at the same path on every node.
prepare_workspace(cell, finder, "/shared/shock_work/output_00785", task_batches=64)
```

Preparation writes physical fields, global normalization, geometry/index,
normals, and original batch IDs. It fails if the directory already exists.
Keep this workspace immutable during a run; prepare again after changing the
measurement implementation, input, or finder settings. `manifest.json` reports its disk size and seed count.
Budget space for both the workspace and final products, and memory for the
preparation input/temporaries and the root's final result. File-backed memory
sharing does not remove shared-filesystem I/O or NUMA bandwidth limits.

For configured execution use `shocktest.launch` as shown in section 1.
The following low-level API examples remain available for custom job wrappers.
Single-node execution (set thread counts before starting Python):

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m shocktest.distributed run --workers 6 \
    --workspace /shared/shock_work/output_00785 --output result_00785.pkl
```

For multiple nodes, install `mpi4py` against the MPI used by the job launcher
and rebuild the native extensions. The optional `mpi` dependency is available
as `pip install -e '.[native,mpi]'`; cluster module installations may require
building mpi4py from source with their `MPICC`. Run inside the scheduler's
allocation and use its supported MPI launch/binding options:

```bash
# Example only: the scheduler must allocate these ranks and their OpenMP cores.
mpirun -np 12 python -m shocktest.distributed run --mpi \
    --workspace /shared/shock_work/output_00785 --output result_00785.pkl
```

Rank zero schedules tasks and merges sparse models; the other ranks compute.
Only rank zero assembles and writes the dense result. The reducer preserves
the original numerical batches and quality ties, independent of task completion
order. Ordinary worker exceptions are propagated to all ranks; incomplete or
duplicate tasks cannot silently produce a partial result. Do not launch the
original single-process runner under `mpirun` without this distributed entry.

Workers retain an exact, bounded Fortran virtual-cell cache across batches.
Spatial ordering groups nearby batches, while keys retain exact coordinates
and sampling widths. The default `--cache-entries 262144` uses about 19 MiB per
worker for persistent cache arrays; `0` disables persistence for comparisons.
This adds to temporary sampling arrays. No seed thinning, fewer rays, or
relaxed physical checks are introduced. Cache hits and queries are recorded in
`result.diagnostics`; speedup must be measured at fixed allocated core count.

The Python APIs are `find_parallel(workspace, workers=...)` and the collective
`find_mpi(workspace, comm=...)`. MPI returns `None` on non-root ranks. A completed
profile result contains the states needed for post-processing without the input
table:

```python
analysis = finder.analyze_result(result, compute_dissipation=True,
                                 build_catalog=False)
```

The CLI accepts `--analysis-options options.json`, containing these keyword
options, to save a `ShockAnalysis` (or `ShockSamples` with `compact=true`) instead
of a `ShockResult`. Existing result fields and analysis consumers are unchanged.
Keep ordinary Python multiprocessing entry points under `if __name__ == "__main__":`.
Input preparation, final assembly, dissipation, and catalog creation are not
MPI-distributed in this release. Completed workspaces are reusable, but
interrupted scans do not yet resume partially completed seed tasks.

The detector does not use a velocity-divergence minimum as its final center,
an intercell normal-dot-product gate, a two/four-crossing switch, or a
temperature floor. Setting the removed `center_normal_cosine`,
`center_plateau_tolerance`, `max_center_steps`, `max_steps`, or `method`
attributes raises an error. `temperature_floor` is removed, including a zero
assignment: reconstructed states are used without a floor. The removed
`validate_mach`, `filter_inconsistent`, `consistency_factor`,
`density_check_max_mach`, `density_saturation_rtol`, and `mach_validation_dtype`
settings also raise migration errors when assigned.

`finder.analyze` combines detection, optional dissipation, and optional catalog
construction. Set `compute_dissipation` and `build_catalog` independently.

Use the following instead of `finder.find(cell)` when you also need dissipation
and a catalog. This example uses Mach tolerance 0.2 and 20 query threads; the
other catalog defaults are listed below.

```python
catalog_options = {
    "connectivity": "touch",   # Allow face, edge, and corner contact.
    "gap_factor": 0.0,         # Do not bridge gaps between cells.
    "mach_tolerance": 0.2,     # Maximum relative Mach difference between neighbors.
    "min_group_size": 3,       # Keep fronts with at least three shock centers.
    "thread": 20,              # Maximum threads for catalog spatial queries.
}
analysis = finder.analyze(
    cell,
    compute_dissipation=True,  # Compute thermalization flux, area, and power.
    build_catalog=True,        # Build fronts and retain their membership labels.
    catalog_options=catalog_options,
    dissipation_options={
        # mu defaults to the mean molecular weight saved by the finder.
        "area_mode": "normal", # Normal-corrected area; "cell" uses dx**2.
    },
    compact=False,             # Return dense ShockAnalysis; True returns ShockSamples.
    compact_options=None,      # Only set when compact=True (see below).
)
result, dissipation = analysis.result, analysis.dissipation
catalog, labels = analysis.catalog, analysis.labels
```

`catalog_options=None` and `dissipation_options=None` use their defaults. Only supply `catalog_options` when `build_catalog=True`.
For a standalone `shock_front_catalog` call, `return_labels=False` returns only the catalog.
`return_labels=True` returns `(catalog, labels)`. Membership means which shock
front each original result record belongs to; it does not contain copies of
positions or other cell measurements.

With `compute_dissipation=False`, the catalog's unavailable area and power are NaN.

Catalog options

| Option | Default | Meaning |
| --- | --- | --- |
| `min_group_size` | 3 | Minimum center count; smaller components are excluded |
| `mach_tolerance` | 0.3 | Maximum `abs(Mi-Mj)/max(Mi,Mj)` on each neighbor pair, range [0,1] |
| `normal_cosine` | 0.5 | Minimum signed unit-normal dot product on each pair |
| `surface_offset_factor` | 0.25 | Normal displacement limited to this times smaller cell width |
| `surface_angle_cosine` | 0.5 | Normal displacement also limited to this times pair distance |
| `connectivity` | `'touch'` | Faces/edges/corners; `'face'` requires positive face overlap |
| `gap_factor` | 0 | Optional extra reach in units of larger cell width; disabled by default |
| `min_mach` | 1 | Finite Mach must be strictly greater than this threshold |
| `require_mach_consistent` | False | Historical archives only; new results do not contain these diagnostics |
| `return_labels` | False | Return `(catalog, labels)` instead of only the catalog |
| `atol`, `rtol` | 1e-9 kpc, 1e-7 | Absolute/relative spatial contact tolerances |
| `normal_tolerance` | 1e-6 | Minimum mean-normal resultant before normalization |
| `thread` | 1 | Maximum spatial query threads; never child processes |
| `chunk_size` | 131072 | Selection/summary batch size |
| `query_chunk_size` | 4096 | Spatial query batch size |
| `max_neighbor_pairs` | 200000 | Candidate-pair batch limit |
| `grouping_method` | `'connected'` | Front grouping; experimental `'flux_persistence'` requires an additional native helper |
| `persistence_dex` | 0.6 | Peak/merge contrast used only by flux-persistence grouping |

These are front-grouping options, separate from the fixed shock detector.
Build the optional persistence helper with `python tools/build_persistence.py` before using
`grouping_method='flux_persistence'`.

Call `analysis.clear()` or `result.clear()` once finished with those arrays;
do not clear a result that is still needed for plotting or galaxy matching.

## 3. Input

`cell` is an AMR **leaf-cell table**, not a dense 3D grid. Supply one-dimensional
arrays with the same length, using these keys and physical units:

| Key | Meaning | Unit |
| --- | --- | --- |
| `("x", "km")`, `("y", "km")`, `("z", "km")` | Cell-center coordinates | km |
| `("dx", "km")` | Cell width | km |
| `("vx", "km/s")`, `("vy", "km/s")`, `("vz", "km/s")` | Velocity components | km/s |
| `("T", "K")` | Gas temperature | K |
| `("rho", "Msol/kpc3")` | Gas mass density | solar masses / kpc³ |
| `"level"` | Integer AMR refinement level | dimensionless |

The geometry must be a common, aligned dyadic leaf mesh with positive widths,
temperatures, and densities. Matching bare field names are also accepted by
the finder. Other supported velocity units are `m/s` and `cm/s`; temperature
must be in K. Supported density units include `g/cc`, `g/cm3`, `g/cm^3`, and
`kg/m3` as well as `Msol/kpc3`.

To supply thermal pressure explicitly, set its exact table key, for example
`finder.thermal_pressure_field = ('thermal_pressure', 'erg/cm3')`. Supported
pressure units are `erg/cm3`, `erg/cm^3`, `dyn/cm2`, and `Pa`. Automatic discovery
is limited to names explicitly denoting thermal pressure (`thermal_pressure`,
`pressure_thermal`, `p_thermal`, `pth`). Otherwise the finder uses the ideal-gas
EOS with the configured mean molecular weight. Temperature and supplied thermal
pressure should describe the same gas; total MHD/CR pressure is not a substitute.

## 4. Output

### Dense detection and dissipation

`find()` returns `ShockResult`, with one row per retained leaf cell.
`analyze(compact=False)` returns `ShockAnalysis`, containing `result`, optional
`dissipation`, `catalog`, and `labels`, plus `counts` and `timings`.

| `ShockResult` field | Meaning |
| --- | --- |
| `mach`, `shock` | Adopted profile pressure-jump Mach and accepted-center mask |
| `mach_estimator` | `'profile_pressure'` for current detections |
| `selected_indices` | Mapping from retained rows to original input rows |
| `center_index`, `upstream_index`, `downstream_index` | Indices in **retained-row space**; endpoint indices identify containing cells; `-1` when unavailable |
| `pos`, `dx`, `level` | Retained geometry; positions have shape `(N, 3)` |
| `center_position` | Fitted subcell positions `(N,3)`; NaN outside measured centers |
| `normal` | Unit temperature-gradient normal from upstream toward downstream |
| `zone_width` | Distance between background support positions, in position units |
| `sampling` | Sparse one-sided fitted states, support geometry, and fit diagnostics, aligned through `sampling['rows']` |
| `diagnostics` | Counts of seeds, rejected profiles, duplicate leaf models, and accepted centers |
| `gamma`, `mean_molecular_weight`, `position_unit`, `temperature_unit`, `density_unit`, `pressure_unit` | Measurement settings and units; reconstructed pressure is in erg/cm3 |

Mach and normals are zero outside detected centers. Additional analysis masks
do not change the stored measurements. For position accuracy,
use `center_position`, not `pos`: the latter remains the containing leaf center.
`zone_width` is a background support span, not a measured physical shock thickness.

`sampling['temperature']`, `['density']`, and `['pressure']` have shape `(K,2)`
for `K` measured centers; columns are upstream/downstream. Both states are
evaluated at the same fitted surface, so `sampling['pos']` has two co-located
positions. `sampling['support_pos']` instead stores the two background support
locations. The upstream/downstream indices refer to leaves containing these
supports, not cells whose center values equal the fitted states. Use the saved
normal or support geometry for orientation.

Other sampling fields include `reference_width`, `density_residual`,
`velocity_residual`, `quality`, `background_orders`, `profile_passes`,
`center_constraint_active`, and `mass_surface_displacement`. The latter records
the displacement from a bounded final surface to its free column-mass solution.
`sampling['converged']` is true for accepted reconstructions, including flagged
bounded solutions; it is not the old temperature-profile convergence criterion.

Additional sparse fields are `velocity` `(K,2,3)` in km/s, `shock_speed_mass`
and `shock_speed_pressure` in km/s, `momentum_residual`, `energy_residual`,
`entropy_margin`, `background_model_count`, `mach_model_range`, and
`center_model_range` (two displacements from the adopted center, position units).
`center_sensitivity` estimates the amplification of background-model errors by
small density contrast. The ranges describe valid models in the selected
constant/nonconstant family; they are not statistical confidence intervals.
`profile_quality_flags` is a `ProfileQualityFlag` bitmask; `quality_clear` means
no listed diagnostic fired. It does not certify an ideal-gas EOS, physical
coordinate conversion, completeness, or a galaxy crossing.

```python
from shocktest import ProfileQualityFlag, galaxy_pressure
s = result.sampling
rows = s['rows']
clear_rows = rows[s['quality_clear']]  # Conservative diagnostic subset.
uncertain = (s['profile_quality_flags'] & int(ProfileQualityFlag.ENTROPY_UNCERTAIN)) != 0
# Gas and galaxy velocities MUST use the same physical frame and km/s.
forcing = galaxy_pressure(s, galaxy_velocity, density_factor=1.0)  # g/cc input
```

Cosmological inputs require verified physical coordinates, velocity conventions,
thermal pressure/EOS, and source masks. A quality-clear flag alone does not
establish completeness or the presence of a physical shock.

New results and NPZ exports do not contain `mach_temperature`, `mach_density`,
`mach_pressure`, duplicate jump-ratio arrays, or the old Mach-consistency flags.
The measured `sampling['temperature']`, `['density']`, `['pressure']`,
`['velocity']`, and `['profile_quality_flags']` remain available for research.
Standalone jump-conversion functions remain available for manual analysis.
Historical pickles retain their stored fields when loaded/resaved; this
compatibility path does not allocate or compute diagnostics in new results.
A stored temperature floor is honored only when reproducing dissipation from
historical temperature-jump results, not for current profile reconstruction.

| `DissipationResult` field | Meaning | Unit |
| --- | --- | --- |
| `flux` | Thermalization power per shock area | erg s⁻¹ kpc⁻² |
| `total` | Power associated with the center's estimated area | erg s⁻¹ |
| `area` | Estimated shock area | kpc² |
| `efficiency` | Thermalization efficiency | dimensionless |
| `sound_speed` | Preshock sound speed | km s⁻¹ |

### Compact storage

```python
samples = analysis.to_compact(profile="science", index_dtype="auto")
samples.save("shock_samples.npz")
restored = shocktest.ShockSamples.load("shock_samples.npz")
analysis.clear()  # samples owns its arrays.
```

`finder.find(cell, compact=True)` returns detections as `ShockSamples`;
`finder.analyze(cell, compact=True)` also includes requested downstream products.
Both compact profiles (`'full'` and `'science'`) preserve `center_position`,
one-sided states, and available profile diagnostics. Compact input and support
indices address the **original input table**. NPZ storage does not require pickle.
Historical pickles retain their original temperature-jump interpretation;
loading them does not recompute detections or add fitted subcell positions.

### Generic fronts from saved detections

`shock_front_catalog(result, dissipation=None, **options)` groups precomputed
shock-center detections.

The return is a NumPy structured array:

```python
front_dtype = np.dtype([
    ('front_id', '<i4'), ('ncell', '<i4'),
    ('center', '<f8', (3,)), ('normal', '<f4', (3,)),
    ('extent', '<f4', (3,)), ('area', '<f8'),
    ('mach', '<f4'), ('diss_rate', '<f8'), ('quality', '<u2'),
], align=False)
```

| Field | Definition and units |
| --- | --- |
| `front_id` | Snapshot-local ID, ordered by minimum original input-cell ID; no temporal identity |
| `ncell` | Number of contributing shock centers |
| `center` | Centroid of fitted `center_position` when available, in kpc; area-weighted, or unweighted if any area is unreliable |
| `normal` | Normalized mean upstream-to-downstream normal; dimensionless; no sign flipping |
| `extent` | Axis-aligned extent in kpc, including half-cell widths on each side |
| `area` | Sum of effective unique-center areas, kpc²; NaN if incomplete |
| `mach` | Area-weighted mean Mach; dimensionless; unweighted if areas are unreliable |
| `diss_rate` | Sum of cell-integrated dissipation rates, erg/s; NaN if incomplete |
| `quality` | uint16 bitmask; multiple conditions can coexist |

Quality constants are exported from `shocktest` and `shocktest.fronts`:

| Bit / constant | Meaning |
| --- | --- |
| 0 / `QUALITY_OK` | No flagged condition |
| 1 / `QUALITY_NO_AREA` | At least one effective area is missing/invalid, or the total overflows |
| 2 / `QUALITY_NO_DISS_RATE` | At least one integrated rate is missing/invalid, or the total overflows |
| 4 / `QUALITY_UNDEFINED_NORMAL` | No valid normals, or their weighted mean cancels |
| 8 / `QUALITY_APPROX_CONNECTIVITY` | Adjacency inferred from saved positions and cell widths |
| 16 / `QUALITY_GAP_BRIDGED` | At least one accepted connection spans a non-contact gap |
| 32 / `QUALITY_PARTIAL_SUMMARY` | Missing aggregate members or numerical overflow |

Connectivity and cell extents use leaf geometry; the reported centroid uses
fitted surface positions. Summed cell-area estimates are not a reconstructed
triangle-mesh surface area or a completeness measurement.

## 5. Usage Python examples

### 5-1. Draw a Mach map and dissipation map

```python
import numpy as np
import matplotlib.pyplot as plt
import shocktest
from shocktest import painter

finder = shocktest.ShockFinder()
finder.min_mach = 1.5
analysis = finder.analyze(cell, build_catalog=False)
result, dissipation = analysis.result, analysis.dissipation

# Use the same explicit selection for both maps. New results default to all detections.
valid = result.shock.copy()
# Optional: choose a profile-quality subset explicitly, then use it for both maps.
# s = result.sampling
# valid[s['rows'][~s['quality_clear']]] = False

extent = painter.map_extent_from_result(result, plane="xy")  # Physical km.
map_options = dict(
    plane="xy",                     # Also supports "xz" and "yz".
    bins=512,                       # Resolution along each projected axis.
    extent=extent,                  # (xmin, xmax, ymin, ymax), in km.
    method="amr",                   # Paint cell footprints, not just cell centers.
    statistic="max",                # Strongest value in each pixel; "mean" is also available.
    min_mach=1.5,
    valid_mach=valid,
    z_center=None, z_width=None,     # Set both in km to select a line-of-sight slab.
    fill_gaps=0,                     # Leave pixels with no measured contribution empty.
)
machmap = painter.make_mach_map(result, **map_options)
dissEmap = painter.make_disspE_map(result, dissipation, **map_options)

KPC_KM = 3.0856775814913673e16
extent_kpc = np.asarray(extent) / KPC_KM
log_flux = np.ma.log10(np.ma.masked_less_equal(np.ma.masked_invalid(dissEmap), 0))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
im0 = axes[0].imshow(np.ma.masked_invalid(machmap), origin="lower",
                     extent=extent_kpc, cmap="viridis")
im1 = axes[1].imshow(log_flux, origin="lower", extent=extent_kpc, cmap="inferno")
fig.colorbar(im0, ax=axes[0], label="Mach number")
fig.colorbar(im1, ax=axes[1], label=r"$\log_{10}[F_{\rm diss}/({\rm erg\ s^{-1}\ kpc^{-2}})]$")
for ax, title in zip(axes, ["Mach map", "Thermal dissipation flux"]):
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", title=title)
fig.savefig("shock_maps.png", dpi=200)
plt.show()
```

`z_center`/`z_width` always refer to the axis perpendicular to the selected
plane. AMR `mean` uses overlap-area weighting; `sum` is an overlap-weighted
projected quantity divided by pixel area, not total dissipated energy.
An empty selection produces empty/NaN map pixels. To include all detected
centers regardless of consistency, pass `valid_mach=result.shock` explicitly.

### 5-2. Build a shock catalog and retrieve its members

For existing `result` and `dissipation` from the same detection run, build a
catalog without rerunning ShockFinder. Keep their original matching row order.

```python
import numpy as np
from shocktest import shock_front_catalog, save_shock_catalog, load_shock_catalog

catalog, labels = shock_front_catalog(
    result,
    dissipation,
    return_labels=True,
    connectivity="touch",
    gap_factor=0.0,
    mach_tolerance=0.2,
    min_group_size=3,
    thread=20,
)

# Select the members of one front, safely handling an empty catalog.
if len(catalog):
    front_id = int(catalog["front_id"][0])
    member_rows = np.flatnonzero(labels == front_id)
    member_positions = result.center_position[member_rows]  # Fitted positions, in result.position_unit.
    member_leaf_positions = result.pos[member_rows]         # Containing leaf centers.
    member_mach = result.mach[member_rows]
    member_flux = dissipation.flux[member_rows]
    input_rows = result.selected_indices[member_rows]  # Rows in the original cell table.
    print("Front:", front_id, "member centers:", len(member_rows))

# Save both summaries and membership; retain the corresponding result separately.
save_shock_catalog("shock_catalog.npz", catalog, labels=labels)
catalog, labels = load_shock_catalog("shock_catalog.npz", return_labels=True)
```

If you already called `finder.analyze(..., build_catalog=True)` in Section 2,
use `catalog, labels = analysis.catalog, analysis.labels` directly and start
with the member-selection block. A label of `-1` means that the record belongs
to no retained front. Membership must stay paired with the same result row order.

For historical results without `center_position`, use `result.pos` and label
those measurements as cell-center positions.

## 6. Reading RAMSES outputs and exporting results

The single-rank RAMSES reader supports the current idealized simulations:
3D, five pure-hydro variables, and gamma=5/3, on either uniform or AMR meshes.
It checks the output layout and converts `unit_l`, `unit_t`, and `unit_d` to
physical units. Multi-rank, RT, and MHD output layouts are not supported by this
reader. OpenMP runs with one MPI rank satisfy the single-rank requirement.

Run detection in a Python script and export the measurements for later analysis:

```python
from shocktest.ramses import load_snapshot, snapshot_finder
from shocktest.profile_analysis import save_profiles, save_slice

cell, meta = load_snapshot("/path/to/run/output_00003")
finder = snapshot_finder(meta)
result = finder.find(cell)
save_profiles("results/profiles.npz", result, meta)
save_slice("results/slice_z.npz", cell, axis="z")
```

The saved NPZ files can be read without the original snapshot or another
detection run. For example, use the following in an analysis notebook:

```python
import numpy as np
import matplotlib.pyplot as plt
from shocktest.profile_analysis import plot_saved_slice

with np.load("results/profiles.npz", allow_pickle=False) as saved:
    mach = saved["mach"].copy()
    quality_clear = saved["quality_clear"].copy()

fig, ax = plot_saved_slice("results/slice_z.npz", field="temperature",
                           profiles="results/profiles.npz")
plt.show()
```

Area weights based on cell width and normal direction are detection-area
proxies. Accuracy distributions conditioned on detected centers do not measure
complete shock-surface coverage. Analytic references, when used for evaluation,
must not enter the detector.

Prepared workspaces use uncompressed binary `.npy` arrays directly for
read-only mmap sharing. No node-local decompression cache is required.
Released v0.5.0 NPY workspaces are accepted when their verified measurement
implementation matches; user input files and manifests are never rewritten.
Legacy compressed workspaces remain readable through the optional staging path.
