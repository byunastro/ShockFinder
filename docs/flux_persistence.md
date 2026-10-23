# Experimental flux-persistence catalog

Select `grouping_method="flux_persistence"` in `shock_front_catalog`, or in
`finder.analyze`'s `catalog_options`. The default `"connected"` method is unchanged.
Build the optional C++ helper first with `python3 tools/build_persistence.py`
(macOS/Linux, C++11 compiler; `CXX` selects the compiler). This helper is separate
from the Fortran modules built by `f2py.sh`. No compilation or child processes
are started by catalog generation.

The method uses DisPerSE-inspired **H0 superlevel persistence**, not the DisPerSE
executable, its full discrete Morse complex, or filament extraction. Existing
shock-center selection, AMR geometry, local normal compatibility and
`mach_tolerance` construct the same adjacency graph as the connected method.
Mach is an auxiliary edge condition, not part of a combined flux/Mach score.
No cluster information or temporal tracking is involved.

Vertices carry log10 of the stored dissipation flux. As the superlevel threshold
falls, the weaker peak dies when its region joins an older peak. Equal peaks
are ordered by original input ID; plateau mergers have zero persistence.
Peaks surviving more than `persistence_dex` seed a marker-controlled maximum
spanning forest. Regions cannot cross existing connected-component boundaries.
The minimum group size is applied after segmentation. IDs are assigned in order
of minimum original input ID, and labels remain aligned with original result rows.

`persistence_dex=0.6` means a peak/merge flux contrast greater than approximately
four. Larger thresholds suppress more splits. This default is an experimental
comparison setting, **not a calibrated significance or an optimal parameter**.
The threshold must be finite and nonnegative; lifetimes <=1e-12 dex are treated
as zero to suppress numerical plateau peaks. Disconnected components retain an
essential peak regardless of threshold.

Every selected center must have finite, strictly positive stored flux; otherwise
the experimental method raises an error. Flux is read using the same alignment
rules as area and integrated rate, including historical ID joins. Compact
inputs use `dissipation_flux`. Flux is never replaced by cell-integrated power.
Missing or zero flux is not silently imputed. The usual 78-byte catalog dtype,
units, quality bits, and optional int32 membership output are unchanged.

Unlike streaming connected components, this experimental implementation stores
all accepted edges and sorting arrays: memory scales as O(N+E), with O(E log E)
sorting cost. `max_neighbor_pairs` bounds candidate batches, not the total graph.
`thread` controls spatial queries; the persistence sweep itself is sequential.

Interpret the result as **dissipation subregions / possible split candidates**.
It can separate regions joined by an artificial weak bridge, but can also split
one physical continuous front with two bright patches even when Mach and normals
are identical. Keep connected catalogs when the aim is whole-front area and
power. Minimum-size filtering can discard pieces of previously retained fronts.
The method does not establish physical shock origins or independent boundaries.

Methodological background: [DisPerSE source and references](https://github.com/thierry-sousbie/DisPerSE).
The helper here is independently implemented; it does not copy or link DisPerSE.
Local comparison scripts, data products and performance measurements are not
part of this branch's source distribution.
