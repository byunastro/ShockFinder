# ShockFinder

Fortran-backed Python shock finder based on the AMR methodology in Skillman et
al. 2008, ApJ 689, 1063.

## Build

```bash
cd shocktest
python3 -m numpy.f2py -c fortran/shockfinder.f90 -m _shockfinder
```

The command writes the compiled extension into `shocktest/`. Run it with the
same Python environment that will import `shocktest`; compiled NumPy/f2py
extensions are Python-version specific.

For large datasets, build with OpenMP:

```bash
cd shocktest
python3 -m numpy.f2py -c fortran/shockfinder.f90 -m _shockfinder \
  --f90flags="-O3 -fopenmp" --opt="-O3 -fopenmp" -lgomp
```

At runtime, set the number of threads before starting Python:

```bash
export OMP_NUM_THREADS=40
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

If the OpenMP build fails on a particular compiler stack, the non-OpenMP build
above still works; it is just single-threaded in the Fortran shock scan.

The package supports NumPy 1.26 or later.

## Use

```python
import shocktest

finder = shocktest.ShockFinder()
finder.maxlevel = 20
finder.minlevel = 13
finder.show_progress = True
finder.boundary = "open"
finder.gamma = 5.0 / 3.0

result = finder.ShockFinder(cell)

mach = result.mach
shock_mask = result.shock

# Free arrays explicitly when running many large regions in one process.
result.clear()
```

`show_progress=True` prints progress while fields are loaded, AMR neighbors are
built, and the Fortran shock scan is running. `progress_interval=0` chooses an
automatic interval of about 5%; set it to an integer number of retained cells if
you want finer or coarser updates.

These call styles are equivalent:

```python
result = finder.find(cell)
result = finder(cell)
result = finder.ShockFinder(cell)
```

For detection, dissipation, and surface grouping together, use the single-pass
pipeline. It reuses the AMR neighbor tables instead of rebuilding them for the
catalog:

```python
analysis = finder.analyze(
    cell,
    compute_dissipation=True,
    build_catalog=True,
    deduplicate=True,
)

result = analysis.result
dissipation = analysis.dissipation
catalog = analysis.catalog
print(analysis.timings)
print(analysis.counts)

analysis.clear()
```

`timings` reports input extraction, neighbor construction, Fortran scan,
dissipation, catalog, detection-total, and end-to-end wall times. The existing
`find()` API still releases neighbor tables immediately after detection and has
unchanged output semantics.

`finder.gamma` is used consistently by the entropy-gradient shock criterion,
the general ideal-gas Rankine-Hugoniot temperature-jump inversion, and the
single-pass dissipation calculation. The default `5/3` preserves the original
monatomic-gas behavior. Physical settings and all required input fields are
validated before the compiled scan; NaN and infinite values are rejected.

Shock-zone walks default to `finder.max_steps = 50` and stop earlier at the
first thermodynamic, convergence, or candidate-zone boundary. Values above 50
remain supported but emit a `RuntimeWarning`. Center finding walks along both
signs of the local temperature-gradient normal and exposes three controls:

```python
finder.max_center_steps = 50
finder.center_normal_cosine = 0.7
finder.center_plateau_tolerance = 1.0e-12
```

The normal is recomputed after each move. Abruptly misaligned moves are
rejected, equal-convergence plateaus select a deterministic representative,
and resolved center paths are cached for reuse by later candidates.

## Input Cell Fields

`cell` is an AMR cell table, not a dense 3D array. The wrapper reads these
fields:

- `cell['x', 'km']`
- `cell['y', 'km']`
- `cell['z', 'km']`
- `cell['dx', 'km']`
- `cell['vx', 'km/s']`
- `cell['vy', 'km/s']`
- `cell['vz', 'km/s']`
- `cell['T', 'K']`
- `cell['rho', 'Msol/kpc3']`
- `cell['level']`

Cells are filtered with:

```python
finder.minlevel <= cell["level"] <= finder.maxlevel
```

Level filtering defines the retained AMR geometry. Temperature and density
limits only control which retained cells may start a shock detection; those
cells remain in the mesh as neighbors and upstream/downstream endpoints.

`dx` is used for AMR geometry and neighbor construction, not as the level
filter.

Inputs are treated as extracted snapshot regions with `boundary="open"`.
Missing cells outside the selected region are not inferred and opposite region
edges are never connected. Other boundary modes are rejected explicitly.

## Output

`ShockResult` has one row per retained input cell:

- `mach`: Mach number at shock centers, zero elsewhere.
- `shock`: Boolean shock-center mask.
- `center_index`: retained-row index of each detected shock center, `-1` elsewhere.
- `upstream_index`: retained-row index used for the preshock state, `-1` elsewhere.
- `downstream_index`: retained-row index used for the postshock state, `-1` elsewhere.
- `selected_indices`: original input-row indices retained after level filtering.
- `normal`: unit shock normal from the retained upstream cell toward the
  downstream cell; zero for non-shock cells.
- `level`: AMR level of every retained cell.
- `zone_width`: upstream-to-downstream center distance in the configured
  position unit; zero for non-shock cells.
- `diagnostics`: counts center/walk step-limit hits, missing neighbors,
  candidate/thermodynamic/convergence exits, invalid jumps, and rejected Mach
  values. These counters are intended for real-snapshot quality comparisons.

Neighbor links are built from AMR cell centers and widths. Same-level face
neighbors are preferred. Fine cells can fall back to coarser face neighbors, and
coarse cells adjacent to refined regions pass the four finer face cells to the
Fortran kernel so gradients can use their face-averaged state.

The default `neighbor_backend="fortran"` uses a compiled open-addressing hash
index and stores finer-face links sparsely. The previous sorted NumPy builder is
available as a correctness reference:

```python
finder.neighbor_backend = "numpy"
```

For repeated runs on identical level-selected geometry, persist the sparse
neighbor tables as memory-mapped NPY files:

```python
finder.neighbor_cache_dir = "/path/to/shockfinder-neighbor-cache"
```

The cache key hashes cell positions, widths, levels, and the sparse table schema;
temperature, density, Mach, and walk settings do not invalidate it.

## Shock Surface Catalog

Face-connected shock centers can be grouped into physical shock surfaces. The
grouping follows same-level and coarse/fine AMR face links and requires both a
similar Mach number and a compatible shock-normal orientation:

```python
from shocktest import pyShockFinder

result = finder.find(cell)
dissipation = pyShockFinder.compute_dissipation(cell, result)
catalog = shocktest.build_shock_catalog(
    result,
    cell=cell,
    dissipation=dissipation,
    mach_tolerance=0.3,
    normal_cosine=0.7,
    deduplicate=True,
)

group_labels = catalog.group_id  # -1 for non-shock cells
for group in catalog.groups:
    print(
        group.mach_peak,
        group.mach_mean,
        group.area,
        group.dissipation_total,
        group.classification,
    )
```

Group means and centroids are surface-area weighted. When a dissipation result
is supplied, `group.area` is in kpc2 and total dissipation is in erg/s. Without
one, area is measured in the square of `result.pos`'s position unit. Grouping
does not change the existing center-only meaning of `result.shock` or
`result.mach`.

When `cell` is supplied, each group also reports its surface-area-weighted
`upstream_temperature`, `upstream_density`, and `external_fraction`. A group is
`external` when at least 80% of its area has upstream temperature at or below
`external_temperature=1e4 K`, `internal` when at most 20% does, and `mixed`
otherwise. These thresholds are configurable with `external_temperature` and
`classification_fraction`. Without `cell`, groups remain `unclassified`.

With `deduplicate=True`, adjacent centers stacked along the shock normal are
collapsed to the strongest-Mach representative for catalog construction.
Tangentially adjacent centers remain separate samples of the same surface. The
mapping is returned as `catalog.center_representative`; suppressed centers have
`group_id == -1`, while the original `result.shock` array remains unchanged.

Catalog threshold stability can be evaluated without rerunning the Fortran
shock finder:

```python
rows = shocktest.analyze_catalog_sensitivity(
    result,
    dissipation=dissipation,
    mach_tolerances=(0.2, 0.3, 0.5),
    normal_cosines=(0.5, 0.7, 0.9),
    duplicate_normal_cosines=(0.7, 0.8, 0.9),
    min_machs=(1.3, 1.5),
)
```

Each row reports the input and representative center counts, group count,
surface area, total dissipation, peak Mach, and area-weighted mean Mach for one
parameter combination.

## Saving Catalogs

The complete catalog can be stored in a versioned, non-pickle NPZ archive and
loaded without the original cell data:

```python
shocktest.save_shock_catalog("shock_catalog.npz", catalog)
catalog = shocktest.load_shock_catalog("shock_catalog.npz")
```

The archive preserves per-cell group labels, center-representative mappings,
variable-length center lists, geometry, physical statistics, classifications,
quality metrics, provenance metadata, and units. Schema-v1 catalogs remain
readable and are conservatively marked as legacy/incomplete. For table
inspection or plotting tools, export one summary row per
group:

```python
shocktest.save_shock_catalog_csv("shock_groups.csv", catalog)
```

CSV is a summary format and does not contain the per-cell mappings required for
a full round trip; use NPZ when the catalog will be loaded back into ShockFinder.

Groups receive deterministic IDs ordered by peak Mach, centroid, and area. Each
group reports six boundary-face flags, `touches_boundary`, computational
`is_complete`, valid-upstream fraction, Mach scatter, normal dispersion,
zone-width statistics, level count, classification confidence, and quality
flags. `is_complete` means the detected group does not touch the extracted
region boundary and all representative centers have valid upstream states; it
does not claim physical completeness outside the supplied region.

Pass run-specific identifiers through `provenance` and inspect catalog health:

```python
analysis = finder.analyze(
    cell,
    provenance={"snapshot": 620, "region": "cluster-core"},
)
summary = shocktest.summarize_catalog_quality(analysis.catalog)
fig, axes = shocktest.plot_catalog_quality(analysis.catalog)
```

## Python Examples

`make_mach_map` and `make_disspE_map` default to `method="amr"`, which paints
each projected AMR shock-cell footprint into the image. This is better for
figure-quality AMR maps than point-binning the cell centers. Use
`method="point"` to recover the older center-binned behavior.

Available statistics are:

- `max`: strongest value touching each pixel.
- `mean`: overlap-area-weighted mean.
- `sum`: overlap-area-weighted projected sum per output pixel area.

You can build mach map and shock dissipated energy map directly from `ShockResult`:

```python
import matplotlib.pyplot as plt
import numpy as np
import shocktest
from shocktest import painter

finder = shocktest.ShockFinder()
finder.minlevel = 15
finder.maxlevel = 20
finder.show_progress = True

result = finder.ShockFinder(cell)
machmap = painter.make_mach_map(result, plane="xy", statistic="max")
diss = pyShockFinder.compute_dissipation(cell, result)
dissEmap = shockpainter.make_disspE_map(result,diss,plane="xz",bins=400,statistic="mean",method='amr')

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.log10(machmap))
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.log10(dissEmap))
plt.show()
```
