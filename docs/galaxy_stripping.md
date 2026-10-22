# Galaxy gas-loss and shock-exposure analysis

`examples.galaxy.classify_galaxy` accepts the combined structured history table,
groups it by persistent `last` branch ID, and returns a dictionary of gas-loss
events, pericenter measurements, shock encounters, galaxy classifications,
sensitivity results, population comparisons, and diagnostic arrays. `hmid` is
only a snapshot-local ID. The event detector operates on gas histories; shock
exposure, pressure, and orbit enter the later mechanism assessment.

The downloaded `dwarf_catalog_620_900.pkl` has physical-kpc positions and
`r50`/`r90`, cosmic-Gyr `t_BB`, and a **stored** `log10(P_ram/k_B)` column.
`vx`/`vy`/`vz` were saved relative to the main cluster velocity. The data
generator defines `ism_flag>=1` as at least 100 selected ISM cells and
`icm_flag>=1` as at least 100 selected ICM cells; the analysis uses these masks
when no explicit resolution mask is supplied. Zero cold-gas and H I masses
have no saved upper limit and are never logged or turned into measured losses.

## Notebook usage

Open `examples/galaxy_stripping_workflow.ipynb` for executable cells covering
one branch, a selected front, the full population, tables, and plots. The
workflow accepts histories already loaded in the notebook; no CLI run or
hard-coded input filename is needed.

```python
from pathlib import Path
import pickle
import pandas as pd
from examples.galaxy_notebook import DEFAULT_UNITS, prepare_saved_shock_search, analyze_saved_galaxies
from examples.galaxy import (
    classify_galaxy, find_pericenter, find_stripped_epoch,
    find_front_affected_galaxies, plot_galaxy_stripping,
)

directory = Path('/Users/byeongyeonghwan/NC_shock')
with (directory / 'dwarf_catalog_620_900.pkl').open('rb') as stream:
    sample_galaxies_history = pickle.load(stream)
with (directory / 'cluster_info.pkl').open('rb') as stream:
    cluster_info = pickle.load(stream)

last = 3751436  # example branch near front 3163
gal_branch = sample_galaxies_history[sample_galaxies_history['last'] == last]
stripped_iout = find_stripped_epoch(gal_branch)
pericenter = find_pericenter(gal_branch, cluster_info, units=DEFAULT_UNITS)

search = prepare_saved_shock_search(
    gal_branch, cluster_info, directory,
    snapshot_range=(700, 900),
    min_front_mach=3.0, min_front_cells=10000,
    fronts=[(785, 3163)],
    options={'pressure_scale': 'log10'},
)
result_catalog = classify_galaxy(
    gal_branch, search.load_product, cluster_info,
    units=DEFAULT_UNITS, options=search.options, make_plots=False,
)
events = pd.DataFrame(result_catalog['gas_loss_events'])
galaxies = pd.DataFrame(result_catalog['classifications'])
front_3163 = pd.DataFrame(find_front_affected_galaxies(
    result_catalog, 785, 3163, include_without_loss=True))
fig, axes = plot_galaxy_stripping(result_catalog, last)

# For all galaxies, using the same saved-output and front filters:
population, population_search = analyze_saved_galaxies(
    sample_galaxies_history, cluster_info, directory,
    snapshot_range=(700, 900), min_front_mach=3.0,
    min_front_cells=10000, fronts=[(785, 3163)],
    output_dir='artifacts/galaxy_stripping_notebook', make_plots=False,
)
```

`find_stripped_epoch` returns the onset snapshot of the largest significant
episode or `None`. The full catalog retains **all** detected episodes for every
selected tracer, including rejected noisy excursions, with significance and
quality flags. `find_pericenter` returns `peri_iout`, `d_peri` in physical kpc,
times in Gyr, and turning-point confidence. The full classification uses all
verified pericenters rather than only the global distance minimum.

By default, the analysis screens gas histories first and reads shock member
cells only around significant loss episodes. The search window covers the
largest configured timing sensitivity window, local cadence, and one adjacent
history output on each side for moving-front interpolation. No gas-loss event
means no shock search for that branch; its exposure remains **unknown**, not
negative. `options={'shock_search_mode': 'all'}` restores a full-output search.
`extra_shock_snapshots=(785,)` also searches that output and neighboring
history outputs for every galaxy, including galaxies with no detected loss.

## Saved shock products

The raw front summary catalog has no member positions. The paired labels,
result, and dissipation arrays are needed for geometry. The saved result
pickles are about 33 GB each; this repository includes a narrow, read-only
protocol-4 memory-map reader for their known numeric layout. It does not
unpickle arbitrary objects. It checks array byte counts, and the extraction
adapter checks member counts and weighted positions against catalog rows.

```python
from examples.saved_shock_pickle import open_saved_shock_snapshot
from examples.galaxy import shock_product_from_catalog

result, dissipation, labels = open_saved_shock_snapshot(
    directory, 785, position_unit='km')
product_785 = shock_product_from_catalog(
    catalog_785, labels, result, dissipation, 785,
    front_ids=selected_front_ids,
    track_origins=checked_cross_output_tracks,
    complete=False,
)
```

`select_nearby_front_ids` filters front bounding boxes for extraction only.
`match_front_tracks` links only mutually nearest, aligned summary fronts in
adjacent outputs; the encounter code then requires local member-patch
correspondence. Snapshot-local front IDs are never assumed persistent. The
saved `zone_width` is a physical length in the same unit as the shock positions
and cell widths; it is converted to kpc before any overlap test. The
fronts do **not** need a merger-origin label. The galaxy-level merger-shock
candidate score uses modeled exposure, timing, gas loss, pressure change,
tracer agreement, and the independently measured two-cluster core passage.
An unattributed front retains `shock_origin_verified=False`.

`examples/run_galaxy_stripping.py` runs the downloaded 783/785/787 products
with bounded memory:

```bash
python -m examples.run_galaxy_stripping \
  /Users/byeongyeonghwan/NC_shock \
  artifacts/galaxy_stripping_620_900 \
  --front 785:3163
```

The runner uses event-window shock search by default, so large saved ranges
do not cause member-cell extraction at unrelated outputs. Use
`--shock-search-mode all` for a complete shock-exposure survey. An explicit
`--front` probes that output for all galaxies even without a gas-loss event.
`run_summary.json` reports the outputs skipped because no branch needed a
shock query. The runner still requires saved product quartets for catalog
outputs within the requested range.
Event-window selection is conditional on observed gas loss, so the
shock-exposed versus non-exposed *loss fraction* is intentionally unavailable
in this mode. Use `--shock-search-mode all` with complete shock coverage for
that population comparison.

### Choose a snapshot range and front properties

The snapshot range is inclusive. The runner discovers catalog outputs inside
it and requires their matching labels, result, and dissipation files. Outputs
without a saved catalog and catalog outputs without matching galaxy/cluster
history are counted separately in `run_summary.json`; they are never treated
as evidence that a galaxy was not exposed. A catalog with an incomplete saved
product quartet causes an explicit error.

```bash
python -m examples.run_galaxy_stripping \
  /Users/byeongyeonghwan/NC_shock \
  artifacts/galaxy_stripping_700_900_strong_large \
  --snapshot-range 700 900 \
  --min-front-mach 3 \
  --min-front-cells 10000 \
  --front 785:3163
```

The front-property cuts are inclusive and combined with **AND**. Mach is the
catalog's reported mean (area-weighted when areas are reliable), `ncell`
counts member shock centers, optional
`--min-front-flux` uses `diss_rate/area` in erg/s/kpc², and optional
`--min-front-diss-rate` uses total integrated dissipation in erg/s. Missing or
invalid measurements cannot satisfy a requested cut. Fronts that pass the
property cuts are then checked for spatial relevance to galaxy positions;
`--front` can include a passing front outside that automatic spatial filter.
It cannot override a failed property cut. `selected_fronts.csv` records each
used front's properties and why it was included; `run_summary.json` records
the requested range and counts at each selection step.

For a catalog already loaded in Python, the same property selection is:

```python
from examples.galaxy import filter_shock_front_catalog

strong_large = filter_shock_front_catalog(
    catalog_785, min_mach=3, min_cells=10000)
```

## Select galaxies near a particular front

The front number in a projected dissipation map is local to its output. Use
both the output and catalog front ID, such as `785:3163`. The `--front` option
above forces a requested front into the extracted subset even if the automatic
bounding-box filter misses it. Once the analysis is saved:

```bash
python -m examples.query_front_galaxies \
  artifacts/galaxy_stripping_620_900 785 3163
```

This writes `front_00785_3163_overlaps.csv` for every galaxy with a sampled
shock-zone overlap or modeled moving-front encounter, and
`front_00785_3163_loss_candidates.csv` for those also showing a significant
gas-loss onset during or shortly after the encounter. It also saves selected
galaxy time-series plots in `diagnostics/front_00785_3163/`. Each row reports
surface distance, gas aperture, Mach, dissipation, timing, tracer agreement,
pressure enhancement, and whether the encounter is a modeled crossing or
only a single-output overlap. The default timing interval is at least 1.5
local snapshot cadences; `--window-gyr` overrides it. A one-output overlap
does not establish a crossing or prove that the front caused the gas loss.

The equivalent Python call is:

```python
from examples.galaxy import find_front_affected_galaxies

rows = find_front_affected_galaxies(result_catalog, 785, 3163)
galaxy_ids = [row['galaxy_id'] for row in rows]
all_overlaps = find_front_affected_galaxies(
    result_catalog, 785, 3163, include_without_loss=True)
```

Only three shock outputs are available. The script marks the extracted
products `complete=False`, so absence of a measured encounter elsewhere does
not support ordinary-RPS attribution. A single-output proximity is not a
confirmed crossing. This limitation belongs in the scientific interpretation
of the output, not just in the plotting legend.

## Detection and classification rules

`StrippingOptions` exposes the log-mass median width, fractional-loss and
peak-loss-rate cuts, maximum episode duration, cadence/gap criteria, tracer
agreement interval, pressure jump/rise factors, association window, encounter
threshold, and merger-phase interval. Only resolved positive mass samples enter
logarithms. Explicit positive upper limits may establish **lower bounds** on
loss; unknown zeros, negatives, missing samples, and unresolved values cannot.
A short missing run bracketed by two resolved masses can establish an
**interval-censored** loss with a measured end-to-end fraction and average
rate. Its onset time remains unresolved, so it is cataloged but cannot by
itself establish a specific stripping mechanism. A terminal unknown gas mass
also prevents a confident no-stripping decision.
A temporary mass spike that falls back to its previous level is rejected as a
stripping episode. Drops that recover within the configurable 0.15 Gyr
`rebound_window_gyr` are rejected even when the low state lasts more than one
snapshot. Onset, peak, end, fraction, rate, duration, and inverse mean
log-loss rate are saved per tracer.

Pressure is used only after gas-loss detection. Isolated one-output pressure
spikes are retained as `P_ram_raw_peak` but excluded from the sustained-rise
score when both neighbors agree; a sudden pressure step needs a
subsequent valid measurement to show that it persists. Missing pressure near
an event limits the mechanism assessment without erasing the gas-loss event.

The merger-shock candidate gate requires a credible modeled front encounter
near or before gas-loss onset, merger-phase context for an unattributed front,
an adequate combined score, and a sudden measured pressure enhancement when
local pressure is available. The ordinary-RPS gate requires a verified
pericenter, adequate saved shock coverage with no relevant encounter, inward
motion, and a smooth measured pressure rise when available. Competing or
unresolved evidence remains mixed. No strong stripping requires sufficiently
sampled gas histories. Scores are evidence indices in [0, 1], **not**
probabilities or causal proof.

The sensitivity catalog varies gas fraction/rate cuts, smoothing width,
association window, and pressure factors one at a time. A classification
unstable across those variants is downgraded to mixed. The population figure
compares exposure groups, category timescales, fractional loss against shock
and pericenter time offsets, and sensitivity counts. All population
comparisons are descriptive.
