# ShockFinder

ShockFinder is a Fortran-backed Python framework for detecting shocks in the intracluster medium
(ICM), measuring Mach numbers and thermal dissipation, groups shock structures,
and supporting galaxy matching and time-resolved exposure measurements. The AMR
shock detector follows the methodology of Skillman et al. (2008, ApJ 689, 1063).

For galaxy gas-loss classification with saved shock fronts, start with the
[notebook workflow](examples/galaxy_stripping_workflow.ipynb) or its
[function reference](docs/galaxy_stripping.md).

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

# OMP settings
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

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
finder.temperature_floor = 1.0e4     # Preshock temperature floor [K] for Mach/sound speed.
finder.min_temperature = None        # Optional seed-cell lower T cut [K]; None disables it.
finder.min_density = None            # Optional seed-cell lower density cut [Msol/kpc3].
finder.max_density = None            # Optional seed-cell upper density cut [Msol/kpc3].
finder.min_mach = 1.0                # Minimum accepted temperature-jump Mach number (>= 1).

# Unit keys used to access the input table; use these units for the full pipeline.
finder.position_unit = "km"          # Positions and cell widths.
finder.velocity_unit = "km/s"        # Velocity components.
finder.temperature_unit = "K"        # Gas temperature.
finder.density_unit = "Msol/kpc3"     # Gas mass density.

# Independent pressure/density-jump Mach checks.
finder.validate_mach = True          # Attach secondary Mach estimates and consistency flags.
finder.filter_inconsistent = False   # True removes failed checks from result.shock only.
finder.consistency_factor = 1.5      # Accept secondary Mach within [M_T/f, f*M_T], f >= 1.
finder.density_check_max_mach = 3.0   # Restrict density consistency checks to weak shocks.

# Memory and progress controls.
finder.show_progress = False         # Print input, neighbor, and shock-scan progress.
finder.progress_interval = 0         # 0: automatic (~5%); otherwise a retained-cell count.

result = finder.find(cell)
print("Detected centers:", result.shock.sum())
```

The adopted Mach number always comes from the temperature
jump. With `filter_inconsistent=False`, validation is diagnostic and does not alter the
shock mask. With filtering enabled, Mach values and endpoint indices remain
available for auditing rejected centers: always select using `result.shock`.

Mach measurement samples the cell **two cells away from the shock center on
each side**, along the center's temperature-gradient normal. If this preliminary
temperature-jump Mach number is **10 or greater**, both rays continue to
**four cells away** and the Mach number is recomputed from those endpoints.
These are individual endpoint temperatures, not averages over two or four cells.
The upstream temperature floor applies to both estimates; `min_mach` is applied
only to the final estimate. Saved endpoint indices, validation ratios, and
dissipation all use the final selected endpoints.

Distances count cells crossed by the AMR ray, excluding the center; across a
refinement boundary they need not equal two/four center-cell widths. Sampling
continues past the old shock-zone stopping conditions. If a required endpoint
is missing, or `max_steps` prevents reaching it, the center is rejected rather
than accepting a shorter measurement. `max_steps` remains a safety cap (default
50); values above four do not increase the measurement distance. The legacy
`candidate_exit`, `thermodynamic_exit`, and `nonconverging_exit` diagnostic
counters remain present and are zero under this sampling rule.

The switch depends on the **measured two-cell Mach**, not an independently known
true Mach: a strong shock underestimated below 10 at this stage will still use
two cells. Four-cell sampling also cannot guarantee recovery if the numerical
transition extends beyond those endpoints.

Level cuts change the retained mesh, so overly restrictive cuts can remove
needed neighbors. Temperature/density cuts affect only detection seeds: other
retained cells remain available as neighbors and endpoints. Center search uses local gradient normals,
relative plateau tolerances, and physical-position tie breaking. AMR sampling
uses the geometry of the contributing cells, including refinement interfaces.

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
        "mu": 0.59,            # Mean molecular weight for the sound speed.
        "area_mode": "normal", # Normal-corrected area; "cell" uses dx**2.
    },
    compact=False,             # Return dense ShockAnalysis; True returns ShockSamples.
    compact_options=None,      # Only set when compact=True (see below).
)
result, dissipation = analysis.result, analysis.dissipation
catalog, labels = analysis.catalog, analysis.labels
```

`catalog_options=None` and `dissipation_options=None` use their defaults. Only supply `catalog_options` when `build_catalog=True`.
`return_labels=False` returns only the catalog.
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
| `require_mach_consistent` | False | Require saved Mach-consistency diagnostics |
| `return_labels` | False | Return `(catalog, labels)` instead of only the catalog |
| `atol`, `rtol` | 1e-9 kpc, 1e-7 | Absolute/relative spatial contact tolerances |
| `normal_tolerance` | 1e-6 | Minimum mean-normal resultant before normalization |
| `thread` | 1 | Maximum spatial query threads; never child processes |
| `chunk_size` | 131072 | Selection/summary batch size |
| `query_chunk_size` | 4096 | Spatial query batch size |
| `max_neighbor_pairs` | 200000 | Candidate-pair batch limit |


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

## 4. Output

### Dense detection and dissipation

`find()` returns `ShockResult`, with one row per cell.
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
| `diagnostics` | Counters for missing neighbors, search limits, and rejected jumps; legacy exit counters are zero |
| `gamma`, `temperature_floor`, `position_unit` | Measurement settings and geometry unit |

Mach and normals are zero outside detected centers. Filtering inconsistent
centers changes the mask, not their stored measurements. `zone_width` measures
the selected two/four-cell sampling span projected on the normal; it is not
automatically a physical interaction width or the full numerical shock thickness.

With validation enabled, additional fields include `mach_temperature` (an alias
of `mach`), `mach_pressure`, `mach_density`, the three jump ratios,
`pressure_check_valid`, `pressure_consistent`, `density_check_valid`,
`density_consistent`, `density_check_applicable`, `mach_consistent`, and
`mach_validation_status`. The latter is a bit mask described by
`shocktest.MachValidationFlag`. Loading reconstructs
`pressure_check_valid`, `pressure_consistent`, `density_check_valid`,
`density_check_applicable`, `density_consistent`, and `mach_consistent` from bits
0, 1, 3, 4, 5, and 7 respectively.

| `DissipationResult` field | Meaning | Unit |
| --- | --- | --- |
| `flux` | Thermalization power per shock area | erg s⁻¹ kpc⁻² |
| `total` | Power associated with the center's estimated area | erg s⁻¹ |
| `area` | Estimated shock area | kpc² |
| `efficiency` | Thermalization efficiency | dimensionless |
| `sound_speed` | Preshock sound speed | km s⁻¹ |


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
    member_positions = result.pos[member_rows]  # In result.position_unit.
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
