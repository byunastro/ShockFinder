# NewCluster saved-input inspection

Inspected on 2026-09-28 without running or changing ShockFinder.

The requested server directory `/storage1/byunkh/NC_shock` does not exist on
this computer. Local copies were discovered under
`/Users/byeongyeonghwan/NC_shock`, using the real filenames
`result_00783.pkl`, `dissipation_00783.pkl`, `catalog_00783.pkl`, and the
corresponding `00785` files. Their flattened directory layout differs from
the example patterns. Snapshot 787 is present but outside 605–785; none of
its contents were opened. Selected adjacent snapshots are therefore 783 and
785. This discovery does not establish that the original host contains only
these snapshots.

## Sizes and node counts

| Snapshot | Retained AMR rows | Accepted shock centers | Result size | Dissipation size | Catalog |
| --- | ---: | ---: | ---: | ---: | --- |
| 783 | 288,595,489 | 2,081,325 | 34,920,056,242 bytes | 11,543,820,010 bytes | 4 bytes, `None` |
| 785 | 288,808,448 | 2,054,795 | 34,945,824,281 bytes | 11,552,338,370 bytes | 4 bytes, `None` |

These six files occupy approximately 86.58 GiB. Wholesale deserialization is
inappropriate. The inspector skips binary pickle payloads, obtains array
metadata from the actual NumPy pickle state, maps only the fields being
examined, and scans the accepted-center mask in 1,000,000-row chunks. The two
compact int64 center-ID lists occupy 33,088,960 bytes together. Mappings are
closed after use. No dense ShockFinder arrays are copied into output files.

The exact requested tree dtype occupies 124 bytes per node. For these two
snapshots alone, its uncompressed size would be 512,878,880 bytes. The full
range has not been measured; a future builder must use disk-backed storage
rather than assume the whole range fits in memory.

## Actual schemas

The pickle protocol is 4. All representative files are complete, with a STOP
opcode and no trailing bytes. The result object type is
`shocktest.core.ShockResult`, a slotted dataclass, rather than a structured
array or dictionary. In the table, `N` denotes that snapshot's retained AMR
row count, including cells that are not accepted shocks.

| Result field | Saved dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `mach` | float64 | `(N,)` | Adopted temperature-jump Mach number |
| `shock` | bool | `(N,)` | Accepted shock-center mask |
| `center_index` | int32 | `(N,)` | Shock-center index in retained-row space |
| `upstream_index`, `downstream_index` | int32 | `(N,)` | Endpoint indices in that retained-row space |
| `selected_indices` | int32 | `(N,)` | Retained-row to original input-row mapping |
| `pos` | float64, Fortran order | `(N,3)` | Cartesian positions |
| `dx` | float64 | `(N,)` | Cell widths, in the position unit |
| `normal` | float64, Fortran order | `(N,3)` | Temperature-gradient shock normal |
| `level` | int32 | `(N,)` | AMR level |
| `zone_width` | float64 | `(N,)` | Endpoint separation projected on the normal |
| `mach_temperature` | float64 | `(N,)` | Alias of the same `mach` payload |
| `mach_pressure`, `mach_density` | float32 | `(N,)` | Secondary jump diagnostics |
| `temperature_ratio`, `pressure_ratio`, `density_ratio` | float32 | `(N,)` | Jump ratios |
| `pressure_check_valid`, `pressure_consistent`, `density_check_valid`, `density_consistent`, `density_check_applicable`, `mach_consistent` | bool | `(N,)` | Physical-validation diagnostics |
| `mach_validation_status` | uint16 | `(N,)` | Validation bit mask |
| `diagnostics` | dictionary | — | Detection counters, listed in `inspection.json` |
| `gamma` | Python float | — | `1.6666666666666667` |
| `temperature_floor` | Python float | — | `10000.0` |
| `position_unit` | Python string | — | `"km"` |

There is **no `shock_id` field**, no embedded snapshot number, no `aexp`,
redshift or physical time, no upstream velocity, and no box-size or coordinate
frame metadata. The sole numeric suffix in each filename identifies the
snapshot: `00783` means integer snapshot 783. All three filenames in each
triplet agree, but their contents cannot independently verify that label.

The dissipation object type is `shocktest.pyShockFinder.DissipationResult`.
Its complete field set is:

| Field | Saved dtype / shape | Repository-defined unit |
| --- | --- | --- |
| `flux` | float64 `(N,)` | erg s⁻¹ kpc⁻² |
| `total` | float64 `(N,)` | erg s⁻¹ |
| `area` | float64 `(N,)` | kpc² |
| `efficiency` | float64 `(N,)` | Dimensionless |
| `sound_speed` | float64 `(N,)` | km s⁻¹, upstream sound speed |

Units above come from the current repository's `compute_dissipation` source;
they are not serialized labels in these files. The saved dissipation objects
contain no identifiers, snapshot metadata, source fingerprint, or row-order
mapping. Their lengths equal those of their associated result files. This
agreement is insufficient to verify an ID join.

Both catalog objects are literally Python `NoneType`, with no dtype, keys,
shocks, IDs, or ordering. They cannot satisfy the requested catalog join.

## Geometry and orientation checks

The saved position unit is explicitly km. The repository reads Cartesian
cell coordinates and `dx` in `result.position_unit`. The physical/comoving
frame, any h convention, coordinate origin, region extraction history, and
the simulation periodic-box size are not saved. Existing examples use open
boundaries for extracted regions; this is not evidence that NewCluster itself
is nonperiodic. Neither cosmological scaling nor periodic wrapping has been
assumed during inspection.

Normals in the repository are normalized temperature gradients directed from
upstream toward downstream. For 1,024 evenly spaced accepted centers in each
snapshot, all normals were finite and unit length to `1e-8`; every sampled
normal had positive projection onto the saved upstream-to-downstream endpoint
separation. No sign correction or absolute cosine was applied. These are
sample checks, not full-tree validation.

| Sample measurement | Snapshot 783 | Snapshot 785 |
| --- | ---: | ---: |
| Mach range | 1.30084–26.9150 | 1.30004–14.9055 |
| Normal norm range | 0.9999999999999998–1.0000000000000002 | 0.9999999999999998–1.0000000000000002 |
| Cell width range, km | 2.63425e15–1.68592e17 | 2.63810e15–1.68839e17 |
| AMR level range | 14–20 | 14–20 |
| Valid position/normal/Mach/width samples | 1,024/1,024 | 1,024/1,024 |
| Positive endpoint-normal projection | 1,024/1,024 | 1,024/1,024 |

All measurement ranges are sample ranges. Full array metadata, Cartesian
sample bounds, payload offsets, and exact scalar fields are in
`inspection.json`.

## ID scope and approved reference convention

Every accepted `center_index` was checked, not merely sampled. It equals the
cell's retained row in its own result object and is unique within that
snapshot. It denotes **one shock-center AMR cell detection**, not an already
aggregated physical shock surface. `selected_indices` is an original input
row mapping, not evidence of persistent material-cell identity.

The user has authorized retaining these existing `center_index` values as
`shock_id`. Their cross-snapshot uniqueness audit found **23,006 repeated
values between 783 and 785**, including 54, 107, 146, 163, and 172. A tree of
these records follows shock detections in physical space; neither AMR IDs nor
individual grid cells can be treated as persistent between outputs.

Snapshot-local `fat` and `son` could be disambiguated with their adjacent
snapshot context, but snapshot-local `first` and `last` cannot safely support
`tree[tree['last'] == shock['last']]`. Different branches can have terminals
with the same local ID. Thus the requested ID equality invariants and global
branch selection cannot all hold under the original local-ID interpretation.

The user explicitly approved preserving `shock_id = center_index` and encoding
references as `node_key = (int64(timestep) << 32) | int64(center_index)`.
All four reference fields contain node keys. Reciprocity consequently compares
node keys, and an isolated node has `first == last == node_key`. The original
center ID is retained exactly in `shock_id`. Encoding bounds are checked and
the required dtype is unchanged.

The user also supplied the exact producer:
`finder.analyze(..., compute_dissipation=True, build_catalog=False,
compact=False)`, followed by dumping the three objects from the same analysis
with one tag. The repository implementation passes that exact result to dense
dissipation calculation. Its implicit retained-row ID is therefore verified
by this producer contract. Every accepted center ID is checked against its
retained-row ID, required dense shapes must agree, and selection is keyed by
those verified IDs. Finite `total == flux*area` is additionally checked for
corruption. Equal row counts alone remain insufficient evidence of a join.

`catalog=None` is the intentional output of `build_catalog=False`. The pipeline
locates and validates these files and records that there are no catalog
records to join. It does not invent groups or rerun detection. Identified
dictionary catalogs are separately supported through ordering-independent
joins on explicit center IDs. Serialized source fingerprints and independent
snapshot tags remain absent; the supplied producer and external metadata table
provide that provenance.

Exact `aexp`, physical time, frame/origin and boundary declarations are still
required. Upstream velocity is absent, so `M*sound_speed` is never used as a
simulation-frame velocity. The first pair uses a bounded spatial search;
later established branches can provide motion predictions. Matching positions
are comoving kpc; output positions are physical kpc. Normals keep their sign.

## Completed validation and remaining execution

All **41 tests** passed: nine inspection tests, sixteen pairwise physical
association tests and sixteen input/branch/output tests. These cover every
requested physical scenario, global assignment conflicts, changing AMR IDs,
cosmological expansion, shuffled catalogs, missing metadata, disconnected
branch labels, corrupted graphs, conflicting box metadata and resource failure
without corrupting a previous main output.

The labelled synthetic four-output example saved 38 nodes, 27 primary links
and 11 branches. All 12 invariants passed and the requested branch selection
worked. A separate CLI run exercised automatic short-trial then full synthetic
range. Matching overlays and branch distributions were visually inspected.
All temporal associations in these demonstration products are explicitly
synthetic.

Complete accepted-record validation of actual snapshots 783/785 passed the
supplied dense producer ID contract and geometry/Mach/normal/width checks.
The configurable extra `mach_consistent` quality cut gives:

| Snapshot | Accepted centers | Consistency failures excluded | Valid matching nodes |
| --- | ---: | ---: | ---: |
| 783 | 2,081,325 | 331,467 | 1,749,858 |
| 785 | 2,054,795 | 323,680 | 1,731,115 |

Every accepted center had finite position, finite nonzero normal, positive
Mach and positive cell width. All remaining nodes have finite positive flux.
Counts and every excluded ID are saved in `local_check/input_validation.json`
and `local_check/invalid_records.csv`. Normals are normalized during compact
loading for temporal matching.

The implementation now includes validated bounded loading, indexed matching,
heuristic confidence, global sparse one-to-one assignment, disk-backed branch
construction, exhaustive invariants, diagnostics and atomic compressed saving.
The main file contains only the exact required `shock_tree` structured array.

Exact time/aexp/frame/box metadata and full server inputs remain unavailable
locally. Therefore a **real core-passage trial, 783→785 association and full
605–785 tree have not been produced**. The exporter reads only RUR headers on
the simulation host; its actual host execution remains untested. See
`README.md` for configuration, score definition, resource guards and execution
commands. Weights/speed bounds and full-size assignment resources remain
uncalibrated. Input files, ShockFinder physics and preexisting workspace edits
are intact.
