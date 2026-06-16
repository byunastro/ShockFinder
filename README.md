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
finder.progress_interval = 0

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

`dx` is used for AMR geometry and neighbor construction, not as the level
filter.

## Output

`ShockResult` has one row per retained input cell:

- `mach`: Mach number at shock centers, zero elsewhere.
- `shock`: Boolean shock-center mask.
- `center_index`: retained-row index of each detected shock center, `-1` elsewhere.
- `upstream_index`: retained-row index used for the preshock state, `-1` elsewhere.
- `downstream_index`: retained-row index used for the postshock state, `-1` elsewhere.
- `selected_indices`: original input-row indices retained after level filtering.

Neighbor links are built from AMR cell centers and widths. Same-level face
neighbors are preferred. Fine cells can fall back to coarser face neighbors, and
coarse cells adjacent to refined regions pass the four finer face cells to the
Fortran kernel so gradients can use their face-averaged state.

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
