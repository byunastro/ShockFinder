# ShockFinder

ShockFinder is a Fortran-backed Python framework for studying how cluster
merger shocks affect galaxies. It detects shocks in the intracluster medium
(ICM), measures Mach numbers and thermal dissipation, groups shock structures,
and supports galaxy matching and time-resolved exposure measurements. The AMR
shock detector follows the methodology of Skillman et al. (2008, ApJ 689, 1063).

## 1. Build

Requirements: Python >= 3.10, NumPy >= 1.26, and a Fortran compiler such as
GNU Fortran.

For GNU Fortran with the Python 3.10/3.11 distutils f2py backend, an OpenMP build is:

```bash
cd shocktest
python -m numpy.f2py -c fortran/shockfinder.f90 -m _shockfinder \
  --f90flags="-O3 -fopenmp" -lgomp
cd ..
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

OpenMP compiler/linker options depend on the compiler and f2py backend; the
command above is not a universal Meson build command. The serial build remains
supported. Set thread variables before starting Python and choose a thread count
appropriate to the available cores and memory.

Run the validation suite after building:

```bash
python -m pytest -q
python -m examples.time_resolved_exposure
```

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

For detection, dissipation, and grouping in one pass, prefer `analyze()`: it
reuses neighbor tables for the catalog instead of reconstructing them.

```python
analysis = finder.analyze(
    cell,
    compute_dissipation=True,         # Thermalization flux, area, and power.
    build_catalog=True,               # Connected shock groups.
    deduplicate=True,                 # Suppress duplicate detections of the same front.
    mach_tolerance=0.3,               # Maximum relative Mach difference for linking centers.
    normal_cosine=0.7,                # Minimum absolute normal dot product for linking.
    duplicate_normal_cosine=0.8,      # Alignment threshold used for duplicate suppression.
    min_mach=None,                    # Catalog cut; None inherits finder.min_mach.
    external_temperature=1.e4,        # Upstream T threshold [K] for external/internal labels.
    classification_fraction=0.8,      # Required area fraction for a single thermal class.
    boundary_margin_cells=0.0,        # Extra boundary-proximity margin in local cell widths.
    minimum_group_centers=2,          # Minimum group size for quality assessment.
    maximum_normal_dispersion=0.3,    # Dispersion threshold for quality assessment.
    provenance={"region": "cluster"}, # User metadata, e.g. snapshot ID and extraction bounds.
    dissipation_options={
        "mu": 0.59,                  # Mean molecular weight for the sound speed.
        "area_mode": "normal",       # Normal-corrected area; "cell" uses dx**2.
    },
    compact=False,                   # False returns dense ShockAnalysis; True: ShockSamples.
    compact_options=None,            # Only supply compact options when compact=True.
)
result = analysis.result
dissipation = analysis.dissipation
catalog = analysis.catalog
print(analysis.counts)
print(analysis.timings)
```

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
`dissipation` and `catalog`, plus `counts` and `timings`.

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

The quantity plotted as `dissEmap` below is a **flux map**, not time-integrated
energy. Integrating flux over time gives a fluence in erg kpc⁻².

### Shock catalog

`ShockCatalog.group_id` labels retained rows (`-1` outside groups), and
`center_representative` maps detections to representative retained rows.
Duplicate suppression affects the catalog, not the original detection mask.
Each `ShockGroup` contains its center indices, size, Mach statistics, area,
dissipation power, centroid, bounds, mean normal, AMR-level range, upstream
properties, thermal classification, and quality flags.

Groups connect face-neighbor shock centers using Mach and normal similarity.
Scalar group means are area weighted. With dissipation supplied, group area is
in kpc² and dissipation power is in erg/s; otherwise area is in squared position
units. Check `area_unit` before using catalog areas.

`external`, `internal`, and `mixed` refer to the upstream-temperature
classification, **not merger origin**. Boundary/completeness flags describe the
extracted region and available endpoints, not completeness of an entire
physical shock surface. Group IDs are local to a snapshot, not temporal tracks.

### Compact results and storage

For large runs that do not need dense map inputs, request shock-only results
directly:

```python
samples = finder.analyze(
    cell,
    compute_dissipation=True,
    build_catalog=True,
    compact=True,
    compact_options={
        "profile": "science",  # "full" (default) preserves all available diagnostics.
        "index_dtype": "auto", # Lossless int32 narrowing where safe; default is "int64".
    },
)
print(len(samples), samples.nbytes)  # Shock rows; column bytes (excludes Python/group overhead).
print(samples["mach"], samples["dissipation_flux"])
samples.save("shock_samples.npz", compressed=True)
restored = shocktest.ShockSamples.load("shock_samples.npz")
```

`ShockSamples` contains `columns`, `metadata`, `groups`, `counts`, and `timings`.
Its index conventions intentionally differ from dense results:

| Compact column | Index space |
| --- | --- |
| `retained_row` | Row in the dense retained-cell result |
| `input_row` | Original input row of this shock center |
| `center_index`, `upstream_index`, `downstream_index` | Original **input** rows, not compact rows |
| `representative_input_row` | Original input row of the representative center |
| `groups[*].center_indices` | Original input rows |

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

### 5-2. Build and save a shock catalog

```python
import shocktest

finder = shocktest.ShockFinder()
finder.min_mach = 1.5
analysis = finder.analyze(
    cell,
    build_catalog=True,
    compute_dissipation=True,
    deduplicate=True,
    mach_tolerance=0.3,
    normal_cosine=0.7,
    provenance={"snapshot": "replace-with-your-snapshot-ID"},
)
catalog = analysis.catalog
print(shocktest.summarize_catalog_quality(catalog))
for group in catalog.groups[:5]:
    print(group.group_id, group.n_centers, group.mach_peak,
          group.dissipation_total, group.classification, group.quality_flags)

# NPZ preserves catalog membership and metadata; CSV contains group summaries.
shocktest.save_shock_catalog("shock_catalog.npz", catalog, compressed=True)
shocktest.save_shock_catalog_csv("shock_catalog.csv", catalog)
loaded_catalog = shocktest.load_shock_catalog("shock_catalog.npz")
```

For an existing detection, the standalone alternative is
`shocktest.build_shock_catalog(result, cell=cell, dissipation=dissipation,
deduplicate=True)`. Its default `deduplicate=False` differs from `analyze()`;
set it explicitly when comparing workflows. The standalone call may rebuild
geometry. Grouping uses the detection mask; to exclude inconsistent centers
throughout catalog construction, set `finder.filter_inconsistent=True` before
running detection. Use `analyze_catalog_sensitivity` to assess sensitivity to
linking thresholds and `plot_catalog_quality` to inspect catalog quality.

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
    shock_front_catalog,
    classify_galaxy_shock_crossing,
    compact_classification_results,
)

KPC_KM = 3.0856775814913673e16
finder = shocktest.ShockFinder()
finder.min_mach = 1.5
finder.filter_inconsistent = True    # Consistent selection for detection and galaxy matching.
analysis = finder.analyze(cell, build_catalog=False)

# A front-sample dictionary, distinct from the connected-group ShockCatalog.
fronts = shock_front_catalog(
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
