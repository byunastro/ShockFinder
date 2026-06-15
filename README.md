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

result = finder.ShockFinder(cell)

mach = result.mach
shock_mask = result.shock
```

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
neighbors are preferred; if absent, a same-or-coarser AMR neighbor is used.
