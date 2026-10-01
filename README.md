# ShockFinder

ShockFinder is a Fortran-backed Python framework for studying how cluster
merger shocks affect galaxies. It detects shocks in the intracluster medium
(ICM), measures Mach numbers and thermal dissipation, groups shock structures,
and supports galaxy matching and time-resolved exposure measurements. The AMR
shock detector follows the methodology of Skillman et al. (2008, ApJ 689, 1063).

## 1. Build

Requirements: Python >= 3.10, NumPy >= 1.26, and a Fortran compiler such as
GNU Fortran. NumPy releases that use the Meson F2PY backend also require Meson
and Ninja in the selected environment.

From the repository root, build all Fortran extensions with **the same Python
executable that runs the analysis**:

```bash
conda activate my-environment
RUN_PYTHON="$CONDA_PREFIX/bin/python"
PYTHON="$RUN_PYTHON" ./f2py.sh
# Use "$RUN_PYTHON" for the analysis job too.

"$RUN_PYTHON" -c 'import sys, numpy, shocktest; from shocktest.core import _shockfinder; from shocktest import _merger_neighbors; print(sys.executable, numpy.__version__, shocktest.__file__, _shockfinder.__file__, _merger_neighbors.__file__)'

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

`f2py.sh` compiles fortran files into an ignored
`shocktest/_f2py_builds/` subdirectory specific to the Python environment,
extension ABI, and installed NumPy version. It verifies both imports and their
resolved paths before reporting success. No `site-packages` copy is needed:
Python must import this repository's `shocktest` package. For a script outside
the repository, add the **repository root** to `sys.path` (or `PYTHONPATH`),
then check `shocktest.__file__` and `_shockfinder.__file__` as above. Old `.so`
files directly inside `shocktest/` are not selected by the package's normal
imports.

Outside conda, set `RUN_PYTHON` to the absolute path of the Python executable
used by the analysis. Re-run the build command after changing the Python
environment, Python version, or NumPy version, including an upgrade from NumPy
1.x to 2.x. Use the same interpreter for the build, validation, and production
run. If `python` and
`python3` resolve to different executables, do not mix them.

The default GNU Fortran flags remain `-O3 -fopenmp -lgomp`; the merger-neighbor
module also uses `-ffp-contract=off` at AMR contact thresholds. Select a
compiler with `FC` (and, if needed, `CC`). For a non-GNU compiler, set
`SHOCKFINDER_F90FLAGS`, `SHOCKFINDER_OPENMP_LIB`, and
`SHOCKFINDER_FP_CONTRACT_OFF_FLAG` to its equivalent optimization, OpenMP, and
floating-point contraction settings. An empty `SHOCKFINDER_OPENMP_LIB` omits
the explicit OpenMP library. Set OpenMP variables before starting Python and
choose a thread count appropriate to the available cores and memory.

The build also passes the Fortran flags through `FFLAGS`, because NumPy 1.26's
Meson F2PY backend does not apply `--f90flags`. Existing `FFLAGS` are retained.

## 2. Use

The following example explicitly sets
**every finder parameter to its default**, with the purpose of each setting
beside the assignment:

```python
import shocktest

finder = shocktest.ShockFinder()

# Geometry: retain only cells within this inclusive AMR-level range.
finder.minlevel = 0                  # Lowest retained AMR level.
finder.maxlevel = 20                 # Highest retained AMR level.
finder.neighbor_backend = "fortran"  # Fast implementation; "numpy" is a reference alternative.
finder.neighbor_cache_dir = None     # Optional directory for reusable geometry/neighbor caches.

# Thermodynamics and candidate selection.
finder.gamma = 5.0 / 3.0             # Ideal-gas adiabatic index (> 1), shared with dissipation.
finder.temperature_floor = 1.0e4     # Preshock temperature floor [K] for Mach/sound speed.
finder.min_temperature = None        # Optional seed-cell lower T cut [K]; None disables it.
finder.min_density = None            # Optional seed-cell lower density cut [Msol/kpc3].
finder.max_density = None            # Optional seed-cell upper density cut [Msol/kpc3].
finder.min_mach = 1.0                # Minimum accepted temperature-jump Mach number (>= 1).

# Shock-zone and center searches.
finder.max_steps = 50                # Maximum upstream/downstream walk steps; > 50 warns.
finder.max_center_steps = 50         # Maximum center-search steps (>= 1).
finder.center_normal_cosine = 0.7    # Minimum normal alignment during center walks [0, 1].
finder.center_plateau_tolerance = 1.e-12  # Relative convergence tolerance; ties use position.

# Unit keys used to access the input table; use these units for the full pipeline.
finder.position_unit = "km"          # Positions and cell widths.
finder.velocity_unit = "km/s"        # Velocity components.
finder.temperature_unit = "K"        # Gas temperature.
finder.density_unit = "Msol/kpc3"     # Gas mass density.

# Independent pressure/density-jump Mach checks.
finder.validate_mach = True          # Attach secondary Mach estimates and consistency flags.
finder.filter_inconsistent = False  # True removes failed checks from result.shock only.
finder.consistency_factor = 1.5      # Accept secondary Mach within [M_T/f, f*M_T], f >= 1.
finder.density_check_max_mach = 3.0   # Restrict density consistency checks to weak shocks.
finder.density_saturation_rtol = 1.e-6  # Tolerance near the strong-shock density-ratio limit.
finder.mach_validation_dtype = "float64"  # "float32" saves memory but changes secondary arithmetic.
finder.thermal_pressure_field = None # Exact thermal-pressure key, or automatic thermal-only lookup.

# Memory and progress controls.
finder.index_dtype = "int64"         # "auto" uses int32 indices when their range safely fits.
finder.show_progress = False         # Print input, neighbor, and shock-scan progress.
finder.progress_interval = 0         # 0: automatic (~5%); otherwise a retained-cell count.

result = finder.find(cell)
print("Detected centers:", result.shock.sum())
```

`finder(cell)` and `finder.ShockFinder(cell)` are compatibility aliases for
`finder.find(cell)`. The adopted Mach number always comes from the temperature
jump. Pressure must pass its consistency check; density contributes where its
check is applicable and is excluded near saturation. With
`filter_inconsistent=False`, validation is diagnostic and does not alter the
shock mask. With filtering enabled, Mach values and endpoint indices remain
available for auditing rejected centers: always select using `result.shock`.

Level cuts change the retained mesh, so overly restrictive cuts can remove
needed neighbors. Temperature/density cuts affect only detection seeds: other
retained cells remain available as neighbors and endpoints. Missing boundary
neighbors are not extrapolated. Center search uses local gradient normals,
relative plateau tolerances, and physical-position tie breaking. AMR sampling
uses the geometry of the contributing cells, including refinement interfaces.

`finder.analyze` combines detection, optional dissipation, and optional catalog
construction. Set `compute_dissipation` and `build_catalog` independently.
With `build_catalog=True`, it calls the same `shock_front_catalog` used for saved
results; pass grouping settings through `catalog_options`. It always stores
original-result-order membership in `analysis.labels`. No main guard is needed
for catalog query threads. Detector neighbor tables are released before the
catalog's bounded spatial search; grouping never rebuilds the detector mesh.
`dissipation_options` configures dissipation. `compact=True` returns ShockSamples;
`compact_options` controls that product's storage profile.

Dissipation inherits `gamma` and `temperature_floor` from detection; conflicting
explicit settings are rejected. `timings` separates input loading, neighbor
construction, scanning, dissipation, catalog construction, and total time.
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


The framework does not include a simulation-specific snapshot reader. Convert
comoving coordinates, scale factors, and code units before constructing this
mapping. All gas and galaxy positions must use the same coordinate frame; unwrap periodic
regions before passing them to the open-boundary detector.

An optional thermal-pressure array may be selected with
`finder.thermal_pressure_field`, including a tuple key if appropriate. Its
units must be consistent between cells; only pressure ratios enter the Mach
check. Automatic lookup recognizes explicitly thermal names (`thermal_pressure`,
`pressure_thermal`, `p_thermal`, `pth`). Otherwise pressure ratios use `rho*T`,
assuming a common mean molecular weight. Total pressure containing magnetic,
cosmic-ray, or turbulent contributions should not substitute for thermal pressure.

Galaxy examples additionally require stable galaxy IDs and `(N_galaxy, 3)`
position arrays in km. Match galaxy IDs between snapshots before constructing
these arrays; row order alone does not establish identity.

## 4. Output

### Dense detection and dissipation

`find()` returns `ShockResult`, with one row per cell retained by the level cut.
`analyze(compact=False)` returns `ShockAnalysis`, containing `result`, optional
`dissipation`, `catalog`, and `labels`, plus `counts` and `timings`.

| `ShockResult` field | Meaning |
| --- | --- |
| `mach`, `shock` | Temperature-jump Mach number and accepted-center mask |
| `selected_indices` | Mapping from retained rows to original input rows |
| `center_index`, `upstream_index`, `downstream_index` | Indices in **retained-row space**; `-1` when unavailable |
| `pos`, `dx`, `level` | Retained geometry; positions have shape `(N, 3)` |
| `normal` | Unit temperature-gradient normal from upstream toward downstream |
| `zone_width` | Endpoint separation projected on the normal, in position units |
| `diagnostics` | Counters for missing neighbors, search limits, exits, and rejected jumps |
| `gamma`, `temperature_floor`, `position_unit` | Measurement settings and geometry unit |

Mach and normals are zero outside detected centers. Filtering inconsistent
centers changes the mask, not their stored measurements. `zone_width` measures
numerical shock broadening; it is not automatically a physical interaction width.

With validation enabled, additional fields include `mach_temperature` (an alias
of `mach`), `mach_pressure`, `mach_density`, the three jump ratios,
`pressure_check_valid`, `pressure_consistent`, `density_check_valid`,
`density_consistent`, `density_check_applicable`, `mach_consistent`, and
`mach_validation_status`. The latter is a bit mask described by
`shocktest.MachValidationFlag`. Unavailable secondary estimates are NaN;
validation fields are `None` when validation is disabled.

| `DissipationResult` field | Meaning | Unit |
| --- | --- | --- |
| `flux` | Thermalization power per shock area | erg s⁻¹ kpc⁻² |
| `total` | Power associated with the center's estimated area | erg s⁻¹ |
| `area` | Estimated shock area | kpc² |
| `efficiency` | Thermalization efficiency | dimensionless |
| `sound_speed` | Preshock sound speed | km s⁻¹ |
| `selected_indices` | Original input-cell IDs for alignment (absent in legacy files) | index |

The quantity plotted as `dissEmap` below is a **flux map**, not time-integrated
energy. Integrating flux over time gives a fluence in erg kpc⁻².

### Generic fronts from saved detections

`shock_front_catalog(result, dissipation=None, **options)` groups precomputed
shock-center detections independently for one snapshot. It does not run
ShockFinder, infer physical origin, filter by cluster geometry, or track fronts
between snapshots. It replaces the former `merger_shock_catalog` API; there is
no compatibility wrapper or merger classification in this path.

New `DissipationResult` objects carry `selected_indices`, identifying original
input cells. The function joins these IDs to the result, even when dissipation
rows are reordered or some IDs are absent. Duplicate IDs are rejected.
**Legacy files without dissipation IDs require `assume_aligned=True`**, after
verifying they came from the same detection output in the same retained-row
order. Matching array lengths alone does not verify alignment. `None` gives
missing area/rate summaries, or uses embedded dissipation columns when `result`
is a `ShockSamples` object. Position units must be explicitly km, kpc, or Mpc.
Saved `area` is in kpc² and `total` in erg/s; `flux` never substitutes for total.
Only unique accepted center records contribute, not all shock-zone cells.

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
| `ncell` | Number of contributing shock centers; int32 overflow is checked |
| `center` | Area-weighted centroid in kpc; unweighted over all members if any area is unreliable |
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

Incomplete area/rate totals are **NaN, never partial sums**. Mach and normal
means use their valid members, with complete area weights when available and
otherwise equal weights; missing members set `QUALITY_PARTIAL_SUMMARY`.
A cancelling mean normal remains NaN and flagged without rejecting the front.

Grouping uses AMR-width spatial queries and bounded candidate-pair batches.
Every accepted pair must pass cell contact, local signed normal compatibility,
tangential surface geometry, and relative Mach compatibility. Curved fronts can
connect through gradual local changes. Sheets unresolved by these tolerances
cannot be distinguished. No cluster metadata or physical-origin classification
is read. Crop boundaries are always open: no periodic links or box metadata.

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
| `require_mach_consistent` | False | Require saved Mach-consistency diagnostics |
| `assume_aligned` | False | Explicit legacy ID-less dissipation row-alignment assertion |
| `return_labels` | False | Return `(catalog, labels)` instead of only the catalog |
| `atol`, `rtol` | 1e-9 kpc, 1e-7 | Absolute/relative spatial contact tolerances |
| `normal_tolerance` | 1e-6 | Minimum mean-normal resultant before normalization |
| `thread` | 1 | Maximum spatial query threads; never child processes |
| `chunk_size` | 131072 | Selection/summary batch size |
| `query_chunk_size` | 4096 | Spatial query batch size |
| `max_neighbor_pairs` | 200000 | Candidate-pair batch limit |

`result.dx` and declared position units are required. `base_cell_size`,
`linking_length`, `neighbor_tables`, and `box_size` options have been removed.
Missing Mach cannot provide a compatible edge; such records can only form
singleton fronts when permitted by minimum size. Missing normals skip the
unavailable normal tests and flag partial summaries. Numeric quality flags do
not establish completeness beyond the crop or physical identity of a front.
The default thresholds are initial settings, not a dataset-calibrated optimum.

`analyze(build_catalog=True)` and standalone grouping use exactly the same
implementation, defaults, and geometry. `catalog_options` accepts the options
above except `return_labels`: analysis always retains labels. Detection settings
still govern which shocks exist in result. The only catalog builder is
`shocktest.shock_front_catalog`, implemented in `shocktest/fronts.py`.
`build_shock_catalog`, `shocktest/catalog.py`, and `examples/shock_catalog.py`
have been removed, along with the old object catalog, duplicate grouping,
thermal classification, and sensitivity builder. Old `analyze` catalog keyword arguments now belong in
`catalog_options` when applicable; retired classification arguments are rejected.

Membership remains outside the 78-byte rows. `analysis.labels` is int32 with one
entry per row of `analysis.result`, and `return_labels=True` provides the same
mapping for standalone calls. Entries equal to a catalog row's `front_id` select
that front's detections from result; -1 means excluded or below minimum size.
Use result.selected_indices to translate selected result rows back to the
original input cell table. Summary rows alone cannot reconstruct membership.

`save_shock_catalog` accepts `labels` separately and stores rows plus membership
in numeric NPZ schema 3. `load_shock_catalog(return_labels=True)` returns both,
raising if labels were not saved. Keep the original result paired with the
archive: labels do not identify a different or reordered result. Old object
archive schemas are explicitly rejected; regenerate from saved detections.
CSV exports contain only summary fields, never membership.

The per-cell helper for galaxy matching is named
`examples.galaxy.shock_front_samples`; it does not build connected catalogs. No executable
usage examples are maintained for the new catalog API.

### Compact results and storage

For shock-only products, set `compact=True` on analyze. `compact_options` accepts
`profile='full'` or `'science'` and `index_dtype='int64'` or `'auto'`.

`ShockSamples` contains `columns`, `metadata`, `groups`, `counts`, and `timings`.
Its `groups` is the same 78-byte structured catalog, with membership in
`columns["group_id"]`. Compact archives use schema 2; legacy archives require
regeneration. Its index conventions differ from dense results:

| Compact column | Index space |
| --- | --- |
| `retained_row` | Row in the dense retained-cell result |
| `input_row` | Original input row of this shock center |
| `center_index`, `upstream_index`, `downstream_index` | Original **input** rows, not compact rows |
| `group_id` | Snapshot-local front ID, or -1; aligned with compact records |

The `science` profile retains Mach, geometry, endpoint indices, consistency
summary/status, dissipation flux/area/power, and available group mappings. It
omits detailed jump diagnostics, endpoint positions, redundant center/mask
columns, sound speed, and efficiency. Omitted columns require a full product or
recomputation to recover. Physical floating-point columns remain float64;
`index_dtype="auto"` changes only exactly representable integer storage.

Use `analysis.to_compact(...)` to compact an existing analysis, or
`result.to_compact(...)` for detection-only data. These create independent
copies; release the original after conversion if no longer needed. Direct
`compact=True` avoids dense secondary-diagnostic and dissipation outputs, but
AMR geometry and neighbor construction still require full retained-cell arrays:
this is not an out-of-core detector. Current map and static galaxy helpers use
dense results, as shown in Section 5.

NPZ is already a binary array container. `compressed=True` applies lossless
compression; `compressed=False` avoids compression work. A custom header/main
binary format is **not currently implemented**. Removing a container header
alone generally saves little compared with retaining only shock rows and
omitting unused columns. Precision reduction is a separate, potentially lossy
choice; it is not required for compact storage.

For reference, one 1,500,820-cell region with 81,686 detected centers produced
compact column sizes of 20.67 MB (`full`/int64), 19.03 MB (`full`/auto), and
9.72 MB (`science`/auto), without a catalog; the last compressed NPZ was 5.94 MB.
These are dataset-specific measurements, not expected compression ratios for
all snapshots. Exact comparisons confirmed preserved retained measurements
for the memory optimizations.

## 5. Usage Python examples

Each example below assumes `cell` has already been loaded. A threshold of
Mach 1.5 is illustrative; choose cuts appropriate to the scientific analysis.

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

# Use the same selection for both maps. This is also the painter's default policy.
valid = result.shock.copy()
if result.mach_consistent is not None:
    valid &= result.mach_consistent

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

### 5-3. Find shock-crossing or shock-nearby galaxies

#### A. Match galaxy trajectories to a fixed shock snapshot

This example uses helpers in [examples/galaxy.py](examples/galaxy.py).
Prepare `galaxy_ids`, `galaxy_pos_prev`, and `galaxy_pos_now`: the two position
arrays must have shape `(N_galaxy, 3)`, be in physical km, and have identical
ID ordering. The helper compares each trajectory segment with a **fixed** shock
front. It uses bounded blocks rather than a full galaxy-by-shock distance array.

```python
import numpy as np
import shocktest
from examples.galaxy import (
    shock_front_samples,
    classify_galaxy_shock_crossing,
    compact_classification_results,
)

KPC_KM = 3.0856775814913673e16
finder = shocktest.ShockFinder()
finder.min_mach = 1.5
finder.filter_inconsistent = True    # Consistent selection for detection and galaxy matching.
analysis = finder.analyze(cell, build_catalog=False)

# A front-sample dictionary, distinct from the numeric connected-front catalog.
fronts = shock_front_samples(
    analysis.result, analysis.dissipation, min_mach=1.5, min_flux=0.0,
)
classification = classify_galaxy_shock_crossing(
    galaxy_pos_prev, galaxy_pos_now, fronts,
    search_radius_km=100.0 * KPC_KM,  # Neighborhood search around the trajectory.
    width_factor=2.0,                # Zone half-width includes 2 * local cell width.
    zone_width_factor=0.5,           # Plus 0.5 * measured numerical shock width.
    memory_budget_bytes=32 * 1024**2,# Budget for matching work blocks, not total process memory.
)
galaxy_ids = np.asarray(galaxy_ids)
print("Crossing candidates:", galaxy_ids[classification["crossed"]])
print("Nearby along trajectory:", galaxy_ids[classification["near_shock"]])
print("Intersect numerical zone:", galaxy_ids[classification["affected_zone"]])

# Current-position proximity only: use a zero-length trajectory.
current_only = classify_galaxy_shock_crossing(
    galaxy_pos_now, galaxy_pos_now, fronts,
    search_radius_km=100.0 * KPC_KM,
    memory_budget_bytes=32 * 1024**2,
)
print("Nearby now:", galaxy_ids[current_only["near_shock"]])

# Save only selected galaxy rows, preserving their original row and stable IDs.
small = compact_classification_results(classification, keep="near_or_crossed")
small["galaxy_id"] = galaxy_ids[small["galaxy_index"]]
np.savez_compressed("galaxy_shock_matches.npz", **small)
```

`crossed` indicates a signed-plane crossing/contact with a tangential geometry
gate. `affected_zone` tests intersection with the estimated numerical zone.
`near_shock` includes either proximity within the search radius or proximity
within the estimated zone half-width, so it is not strictly a fixed-radius cut.
`distance_to_shock` is the segment-to-selected-center distance, not simply the
normal distance to an infinite plane. Outputs also include `nearest_mach`,
`nearest_flux`, and `nearest_shock_row` (a dense retained-row index; `-1` if none).
The selection favors a zone-intersecting patch when available. Check missing
indices before using them to index arrays.

The fixed-front approximation does not detect a translating shock crossing a
stationary galaxy. Nor does geometric proximity alone establish a merger origin
or quantify energy actually absorbed by the galaxy. Use temporal tracking for
cumulative exposure.

#### B. Track moving shocks and integrate galaxy exposure

A complete executable example is provided in
[examples/time_resolved_exposure.py](examples/time_resolved_exposure.py):

```python
from examples.time_resolved_exposure import run_example

attributions, interval, history = run_example()
print(attributions[1].status)
print(interval.records[0].crossing_times_s)
print(history.summaries[42])
```

It moves a shock from 100 to 400 kpc over 100 Myr past a stationary galaxy at
250 kpc, giving a crossing at 50 Myr. To apply this workflow to snapshots:

1. Construct a `ShockFrame` for each snapshot, directly or with
   `ShockFrame.from_samples()`. Supply time in seconds, positions/radii/half-widths
   in km, normals, Mach numbers, fluxes, and a documented `tracking_source`.
   Supply persistent `patch_id` and `surface_id` values from a tracking method;
   snapshot group IDs and AMR row indices are not persistent IDs.
2. Define `MergerEvent` objects from independent merger-tree or orbital evidence,
   with event times, center, axis, radial range, and an evidence `source`.
   `attribute_merger_shocks(previous, current, events)` applies geometry and
   propagation constraints and returns candidate, ambiguous, excluded, or
   unattributed statuses. Candidate association does not confirm causality.
3. Call `integrate_galaxy_exposure(galaxy_ids, galaxy_pos_prev, galaxy_pos_now,
   previous, current, attributions=attributions)` for each adjacent snapshot pair.
   Add each returned interval to an `ExposureAccumulator`.

Temporal integration assumes linear patch/galaxy motion and flux interpolation
between matched frames; strongly changing normals are rejected. Inspect interval
diagnostics for unmatched patches and rejected pairs. Missing tracks are not
proof of zero physical exposure. Choose patch radii and interaction half-widths
explicitly; numerical shock width alone is not a calibrated galaxy interaction
scale.

Accumulated summaries include crossing counts, exposure duration, peak Mach,
fluence (`fluence_erg_kpc2`), and candidate-merger fluence. Overlapping patches of
one surface use the maximum flux rather than double counting; separate surfaces
contribute additively. Fluence is incident shock exposure per area, not the
energy deposited inside a galaxy. Converting it to galaxy heating requires a
separate interaction/coupling model.
