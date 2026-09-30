# shockTree

Construct primary 1:1 temporal branches of **saved shock-center AMR cell
detections**. ShockFinder detection, Mach estimation and dissipation physics
are neither rerun nor changed. These branches track detections in physical
space; AMR grid cells and IDs are not persistent material cells. The output
does not aggregate cells into physical shock surfaces.

The actual local inputs were inspected before implementation. See
[INSPECTION_REPORT.md](INSPECTION_REPORT.md) for schemas, units, counts, ID
collisions and execution limits. Complete array metadata are in
`inspection.json`; real local filenames are in `input_manifest.csv`.

## Configuration

The requested server defaults and **exact dtype** are in `config.py`.
`config.example.json` exposes matching and resource settings. Relative JSON
paths resolve against the JSON's directory. Input/output paths, snapshot
bounds and metadata/manifest paths also have CLI overrides.

Use Python 3.10+ with NumPy, SciPy and Matplotlib (`requirements.txt`). Run
commands from the ShockFinder repository root. Verification used
`/opt/anaconda3/envs/universe/bin/python` with NumPy 2.0.0, SciPy 1.15.3 and
Matplotlib 3.9.2.

Temporal matching requires these explicit inputs:

- `metadata_path`: CSV with `timestep,aexp,time_gyr` for every selected output.
  Ages must increase. `snapshot_metadata.example.csv` contains a header only;
  example redshifts and guessed time spacing are never substituted.
- `coordinate_frame`: `physical` or `comoving` for saved positions.
- `coordinate_origin`: `same_simulation_origin` or `common_unwrapped_origin`.
  Per-snapshot recentering needs a separate coordinate adapter.
- `periodic`: true or false. If true, supply three comoving kpc box lengths
  using `box_comoving_kpc` or CSV columns
  `box_x_comoving_kpc,box_y_comoving_kpc,box_z_comoving_kpc`.
- A defensible `matching.max_speed_kms`, weights, gates and score threshold.
  Exposed defaults are starting settings tested on synthetic cases; they have
  not been calibrated on NewCluster.

Frame, origin and periodicity start as null because they are not serialized.
Missing declarations or exact metadata stop a real build before saving.

## Approved IDs and verified joins

The user approved preserving `center_index` in `shock_id`, with globally
unique references:

```python
node_key = (np.int64(timestep) << 32) | np.int64(shock_id)
```

`shock_id` remains snapshot-local. `fat`, `son`, `first`, and `last` contain
node keys. Bounds `0 <= timestep < 2**31`, `0 <= shock_id < 2**32` are checked.
`model.node_key` and `model.decode_key` implement the convention. The approved
reciprocal and isolated-node checks consequently use keys:

```python
parent['son'] == node_key(child['timestep'], child['shock_id'])
child['fat'] == node_key(parent['timestep'], parent['shock_id'])
isolated['first'] == isolated['last'] == node_key(isolated['timestep'], isolated['shock_id'])
```

This explicitly replaces equality against bare local IDs. The dtype is
unchanged. Missing links are `-1`, with NaN scores; reciprocal accepted scores
are written from the same float64 value. `last` selection is unambiguous.

The user-supplied producer calls `analyze(..., compute_dissipation=True,
build_catalog=False, compact=False)` and dumps all three objects from that
same analysis with one tag. Dense dissipation therefore uses an **implicit
retained-row ID** from the exact result supplied to its calculation. The
loader verifies every accepted `center_index` equals that retained-row ID,
checks required dense shapes, and verifies finite `total == flux*area` as an
additional corruption check. It selects by verified IDs. Equal lengths alone
are not the evidence for joining. The config names this supplied contract
`analyze_dense_same_call`; other producers need an explicit adapter.

Catalog files are still located and opened. `None` is expected from
`build_catalog=False`, so there are no catalog records to join. Parameters
such as catalog deduplication are inactive in this producer. With an
identified dictionary catalog, set `catalog_mode="identified"`: a unique
`shock_id` array must cover the detected centers and is joined by ID regardless
of ordering. Other catalog schemas fail clearly.

Discovery resolves nested/flattened names and differing suffix widths.
Ambiguous tags, duplicates and incomplete triplets fail. Arbitrary filenames
can use `manifest_path`/`--manifest-path`, with CSV columns
`timestep,result_path,dissipation_path,catalog_path`. Manifest paths resolve
against its directory. Producer tags plus the metadata table establish
snapshot identity; any embedded labels must agree. Previous/next use the
sorted selected list, including empty snapshots, rather than integer increments.

## Execution on the simulation host

Inspect and validate saved inputs first; preflight needs no physical times:

```bash
python -m shockTree.inspect_inputs --input-dir /storage1/byunkh/NC_shock
python -m shockTree.pipeline --config shockTree/config.example.json --preflight
```

The schema inspector can exit 2 after writing its report when external physics
metadata remain unresolved. Preflight checks every accepted record and writes
counts and excluded IDs without temporal matching.

Export exact metadata on the NewCluster host with RUR. This reads headers
only, using the NC conventions established by the repository's existing
exporter for `params['z']`, `params['age']` and `unit['kpc']`. Declare how saved
positions were produced. For example, **if** they use a common physical
simulation origin and need periodic wrapping:

```bash
python -m shockTree.export_metadata \
  --sim-repo /storage7/NewCluster2 \
  --input-dir /storage1/byunkh/NC_shock \
  --output shockTree/snapshot_metadata.csv \
  --coordinate-frame physical \
  --coordinate-origin same_simulation_origin \
  --periodic yes
```

For a verified common unwrapped region, use that origin and `--periodic no`.
Periodic box lengths use `boxlen/unit['kpc']/aexp`, or
`--box-comoving-kpc LX LY LZ`. The exporter writes provenance JSON; copy its
frame/origin/periodicity declarations into the pipeline JSON. Actual RUR export
has not been run here because simulation headers are unavailable locally.

Test a short **measured** core-passage interval first. The following bounds
are examples; choose actual core passage and available outputs from metadata:

```bash
python -m shockTree.pipeline --config shockTree/config.example.json \
  --snapshot-start 708 --snapshot-end 712 \
  --output-path /storage1/byunkh/NC_shock/shock_tree_trial.npz \
  --short-interval
```

Inspect overlays, match fractions, score/residual distributions and rejected
gates to adjust uncalibrated settings. Standard full execution first builds
and validates the supplied short interval, then processes the configured range:

```bash
python -m shockTree.pipeline --config shockTree/config.example.json \
  --trial-start 708 --trial-end 712
```

The trial must contain at least two available selected snapshots. It writes a
separate `_trial.npz`. Graph validation does not calibrate scores or prove
physical identity. Preconditions/resource failures preserve the prior main
output; a successful validated save atomically replaces the configured path.

## Matching and confidence

Saved km/kpc/Mpc positions and widths are converted to **comoving kpc** for
matching, with physical Gyr ages. This removes coordinate expansion of a
stationary comoving detection. Output `x,y,z` are **physical kpc** at each
node's scale factor. Normals are normalized without sign flips, preserving
the verified upstream-to-downstream convention; no absolute cosine is used.

The first pair has no velocity history and searches near the original position
with a speed/time bound and local-cell allowance. Later, sufficiently confident
branches predict linear comoving motion with configurable smoothing and
uncertainty. A separate total-displacement gate always applies. Upstream
velocity is absent, so `M*sound_speed` is never used as simulation-frame velocity.

`cKDTree` radius searches use descendant cell-size bins. Minimum-image wrapping
applies only when declared periodic. The conservative speed distance is
`max_speed_kms*KMS_TO_KPC_GYR*dt/min(a_i,a_j)`; local allowance is
`cell_slack*max(dx_i_comoving,dx_j_comoving)`. Established predictions reduce
the residual speed allowance by `prediction_uncertainty_fraction`. A search
reaching half the periodic box stops because winding is ambiguous.

Gates reject excessive residual/traveled distances, signed normal changes,
Mach ratios and positive dissipation ratios before assignment. Features are:

```text
Cpos = prediction_residual / allowed_residual
Cn   = 1 - dot(unit_normal_i, unit_normal_j)
CM   = abs(log(Mj)-log(Mi)) / mach_log_scale
CE   = abs(log(Ej)-log(Ei)) / dissipation_log_scale
C    = sum(w * feature) / sum(available weights)
```

Dissipation contributes only when both values are finite and positive. Missing
features omit their cost weight. Default `E=flux` (erg/s/kpc²) avoids artificial
changes from AMR cell area; `total` (erg/s) is configurable. Dissipation is not
stored in the tree.

The bounded association score is:

```text
score = exp(-C/score_temperature)
        * [(1-margin_weight) + margin_weight*margin]
        * mutual_ranking_factor
        * available_weight / configured_weight
margin = clip(min(alternative_parent_cost-C,
                  alternative_child_cost-C) / margin_scale, 0, 1)
```

No alternative contributes margin 1; equal-cost alternatives contribute zero.
Forward–backward consistency means the edge ranks first independently among
its parent's descendants and its child's progenitors. Mutual first choices
get factor 1; others get `nonmutual_score_factor`. This is an explicit ranking
check, not an independently measured trajectory or calibrated probability.
Cost includes spatial, normal, Mach and dissipation consistency; missing
features also reduce confidence through the available-weight fraction.

Global sparse minimum-weight bipartite assignment uses real cost `1-score`
and private unmatched columns. Tiny positive offsets retain perfect-score
edges in the sparse solver. Missing/gated edges are absent. The solver chooses
a globally optimal confidence combination with one parent and one child per
node. A conflicting local best may lose to the better global combination;
discarded eligible alternatives are diagnosed separately. Split/merge flags
identify competing detection links, not confirmed physical front events.

## Memory, validation and diagnostics

Only two compact snapshots are retained. A mask-only counting pass allocates
a disk-backed tree and small branch state. Pickle payloads are read in bounded
row chunks, with mappings immediately closed. Forward/backward snapshot passes
finalize branch endpoints using sorted per-snapshot ID indexes. Output is
sorted by timestep and original shock ID.

Nonfinite positions, zero/nonfinite normals, invalid Mach or widths are
excluded and recorded in `invalid_records.csv`. By default, saved
`mach_consistent=False` is also excluded; that extra cut is configurable.
Nonpositive dissipation removes only that matching feature. The minimal dtype
has no validity flag and requires unit normals, so invalid geometry is reported
outside the tree.

No dense pair matrix is formed. Query subdivision and explicit edge-count
limits bound memory; exceeding a limit stops rather than silently discarding
candidates. Highly sampled fronts can have many compatible neighbors. Tune
physical bounds on the real trial and raise budgets only within available RAM.
Full-range time/resource requirements remain unmeasured. Staging and final
output coexist on disk, so conservative free-space checking includes an
incompressible output bound.

The main NPZ contains **only** the structured array named `shock_tree`.
KD-trees, candidate edges and compact snapshots are released after each pair;
temporary disk-backed arrays are removed. Sidecars contain snapshot counts,
match fractions, full score/residual distributions, rejected reasons,
score-ranked secondary candidates, low-confidence branch counts/samples,
length/lifetime histograms, selected XY overlays, configuration and validation.
Diagnostic tables are explicitly capped and mark truncation; histogram/event
counts are exhaustive. All excluded IDs are retained in their separate table.

Validation checks all 12 invariants under the approved reference convention:
reference existence in adjacent selected outputs, reciprocals and equal scores,
finite unit normals, ordering, root/terminal anchors, and connected labels.
Every edge increases topological snapshot layer and physical time, explicitly
excluding temporal cycles. Every root/terminal identifies itself, preventing
disconnected chains from sharing endpoints. Endpoints and lifetimes refer only
to the selected range.

## Loading and verification

```python
import numpy as np
with np.load('/storage1/byunkh/NC_shock/shock_tree.npz', allow_pickle=False) as f:
    shock_tree = f['shock_tree']
shock = shock_tree[index]
branch = shock_tree[shock_tree['last'] == shock['last']]
branch = np.sort(branch, order='timestep')
normal = shock['n']  # shape (3,)
```

```bash
python -m unittest shockTree.test_inspection shockTree.test_matching shockTree.test_pipeline -v
python -m shockTree.synthetic_demo
```

41 tests passed, covering every requested physical case, global conflicts,
cosmological expansion, changed AMR IDs, shuffled catalogs, missing metadata,
disconnected labels, conflicting box metadata, resource failure and corrupted
graphs. The labelled synthetic demo saved
38 nodes, 27 links and 11 branches, with all invariants passing. A CLI run also
verified automatic short-trial followed by full synthetic range. Example
archives and plots are in `demo_output/`.

For actual local 783/785 files, full accepted-record checks passed the supplied
dense producer ID contract and finite position/normal/Mach/width checks.
After the configurable consistency cut, valid counts are 1,749,858 and
1,731,115. Results are in `local_check/input_validation.json` and
`local_check/invalid_records.csv`. Actual time/aexp/frame/box metadata and full
server inputs are unavailable here; the real core-passage match and 605–785
`shock_tree.npz` therefore remain to be run on the simulation host.
