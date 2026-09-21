"""Moving planar shock patches, evidence-based attribution, and exposure proxies.

Positions/radii are physical km, time is seconds, and flux is erg/s/kpc2.
Persistent patch and surface IDs must come from an auditable tracking procedure;
AMR row numbers and per-snapshot catalog group IDs are not persistent IDs.
"""
from dataclasses import dataclass, field
import numpy as np
from .spatial import moving_patch_blocks


def _ids(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a one-dimensional integer array')
    if np.any(values < 0) or (values.dtype.kind == 'u' and np.any(values > np.iinfo(np.int64).max)):
        raise ValueError(f'{name} must fit nonnegative int64')
    return values.astype(np.int64, copy=True)


def _array(values, shape, name, *, positive=False, nonnegative=False):
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    if positive and np.any(values <= 0) or nonnegative and np.any(values < 0):
        raise ValueError(f'{name} has invalid negative/zero values')
    return values.copy()


@dataclass(slots=True)
class ShockFrame:
    time_s: float
    patch_id: np.ndarray
    surface_id: np.ndarray
    pos_km: np.ndarray
    normal: np.ndarray
    radius_km: np.ndarray
    half_width_km: np.ndarray
    mach: np.ndarray
    flux: np.ndarray
    tracking_source: str

    def __post_init__(self):
        if not np.isfinite(self.time_s):
            raise ValueError('time_s must be finite')
        if not self.tracking_source.strip():
            raise ValueError('tracking_source must describe how persistent IDs were assigned')
        self.patch_id = _ids(self.patch_id, 'patch_id')
        self.surface_id = _ids(self.surface_id, 'surface_id')
        n = len(self.patch_id)
        if len(np.unique(self.patch_id)) != n or len(self.surface_id) != n:
            raise ValueError('patch IDs must be unique and surface IDs must align')
        self.pos_km = _array(self.pos_km, (n, 3), 'pos_km')
        self.normal = _array(self.normal, (n, 3), 'normal')
        norm = np.linalg.norm(self.normal, axis=1)
        if np.any(norm == 0):
            raise ValueError('normals must be nonzero')
        self.normal /= norm[:, None]
        self.radius_km = _array(self.radius_km, (n,), 'radius_km', positive=True)
        self.half_width_km = _array(self.half_width_km, (n,), 'half_width_km', nonnegative=True)
        self.mach = _array(self.mach, (n,), 'mach', positive=True)
        if np.any(self.mach < 1):
            raise ValueError('mach must be at least one')
        self.flux = _array(self.flux, (n,), 'flux', nonnegative=True)

    @classmethod
    def from_samples(cls, samples, *, time_s, patch_id, surface_id,
                     radius_km, half_width_km, tracking_source):
        """Convert compact products after assigning persistent tracking IDs.

        Widths are explicit: numerical shock broadening is not a physical
        interaction duration. Callers must choose and test their exposure scale.
        """
        if samples.metadata['position_unit'] != 'km':
            raise ValueError('exposure requires physical positions in km')
        return cls(time_s, patch_id, surface_id, samples['pos'], samples['normal'],
                   radius_km, half_width_km, samples['mach'],
                   samples['dissipation_flux'], tracking_source)


def _matched(previous, current, normal_cosine):
    if current.time_s <= previous.time_s:
        raise ValueError('frame times must increase')
    if not 0 <= normal_cosine <= 1:
        raise ValueError('normal_cosine must be in [0, 1]')
    ids, left, right = np.intersect1d(previous.patch_id, current.patch_id, return_indices=True)
    if np.any(previous.surface_id[left] != current.surface_id[right]):
        raise ValueError('a persistent patch cannot change surface_id within an interval')
    alignment = np.sum(previous.normal[left] * current.normal[right], axis=1)
    accepted = alignment >= normal_cosine
    diagnostics = {
        'matched_patches': int(np.count_nonzero(accepted)),
        'unmatched_previous': len(previous.patch_id) - len(ids),
        'unmatched_current': len(current.patch_id) - len(ids),
        'normal_rejected_patch_ids': ids[~accepted].tolist(),
        'tracking_sources': [previous.tracking_source, current.tracking_source],
        'motion_model': 'linear translation; fixed mean normal; minimum endpoint radius and half-width',
    }
    return left[accepted], right[accepted], diagnostics


@dataclass(frozen=True, slots=True)
class MergerEvent:
    event_id: str
    start_s: float
    end_s: float
    center_km: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius_min_km: float
    radius_max_km: float
    source: str
    min_alignment: float = .7
    min_outward_speed_kms: float = 0.
    max_outward_speed_kms: float = 1.e5

    def __post_init__(self):
        values = [self.start_s, self.end_s, self.radius_min_km, self.radius_max_km,
                  self.min_alignment, self.min_outward_speed_kms, self.max_outward_speed_kms]
        if not np.all(np.isfinite(values)) or self.end_s <= self.start_s:
            raise ValueError('invalid merger event time window or thresholds')
        if not self.event_id or not self.source.strip():
            raise ValueError('merger events require an ID and independent evidence source')
        _array(self.center_km, (3,), 'event center')
        axis = _array(self.axis, (3,), 'event axis')
        if np.linalg.norm(axis) == 0 or not 0 <= self.min_alignment <= 1:
            raise ValueError('invalid event axis/alignment')
        if not 0 <= self.radius_min_km < self.radius_max_km:
            raise ValueError('invalid radial shell')
        if not 0 <= self.min_outward_speed_kms < self.max_outward_speed_kms:
            raise ValueError('invalid outward speed interval')


@dataclass(frozen=True, slots=True)
class MergerAttribution:
    surface_id: int
    status: str
    event_ids: tuple[str, ...]
    evidence: dict


def attribute_merger_shocks(previous, current, events, *, excluded_surfaces=None,
                            minimum_area_fraction=.8, normal_cosine=.99):
    """Conservative candidate attribution using independent merger events.

    Requires temporal overlap, radial shell, merger-axis location and normal
    alignment, and measured outward propagation. Fractions weight patch disks
    by radius squared; overlapping disks are samples, not an exact surface area.
    Known feedback/accretion surfaces can be excluded with documented reasons.
    This establishes compatibility with an event, never causal confirmation.
    """
    if not 0 < minimum_area_fraction <= 1:
        raise ValueError('minimum_area_fraction must be in (0, 1]')
    events = tuple(events)
    if len({event.event_id for event in events}) != len(events):
        raise ValueError('merger event IDs must be unique')
    excluded_surfaces = {} if excluded_surfaces is None else excluded_surfaces
    left, right, diagnostics = _matched(previous, current, normal_cosine)
    output = {}
    dt = current.time_s - previous.time_s
    for surface in np.unique(current.surface_id):
        surface = int(surface)
        all_rows = np.flatnonzero(current.surface_id == surface)
        pair = current.surface_id[right] == surface
        a, b = left[pair], right[pair]
        denominator = np.sum(current.radius_km[all_rows]**2)
        fractions = {}
        if surface in excluded_surfaces:
            if not str(excluded_surfaces[surface]).strip():
                raise ValueError('excluded surfaces require a reason')
            output[surface] = MergerAttribution(surface, 'excluded', (), {'reason': str(excluded_surfaces[surface])})
            continue
        for event in events:
            # Require the whole interval to be inside the event's supplied window.
            if previous.time_s < event.start_s or current.time_s > event.end_s:
                continue
            radial = current.pos_km[b] - np.asarray(event.center_km)
            distance = np.linalg.norm(radial, axis=1)
            direction = np.divide(radial, distance[:, None], out=np.zeros_like(radial), where=distance[:, None] > 0)
            axis = np.asarray(event.axis) / np.linalg.norm(event.axis)
            speed = np.sum((current.pos_km[b] - previous.pos_km[a]) * direction, axis=1) / dt
            valid = ((distance >= event.radius_min_km) & (distance <= event.radius_max_km)
                     & (np.abs(direction @ axis) >= event.min_alignment)
                     & (np.abs(current.normal[b] @ axis) >= event.min_alignment)
                     & (speed > event.min_outward_speed_kms) & (speed <= event.max_outward_speed_kms))
            fractions[event.event_id] = float(np.sum(current.radius_km[b[valid]]**2) / denominator)
        matches = tuple(sorted(key for key, fraction in fractions.items() if fraction >= minimum_area_fraction))
        status = 'candidate' if len(matches) == 1 else 'ambiguous' if matches else 'unattributed'
        output[surface] = MergerAttribution(surface, status, matches, {
            'area_fractions': fractions, 'threshold': minimum_area_fraction,
            'sources': {e.event_id: e.source for e in events},
            'tracking': diagnostics, 'causal_confirmation': False,
        })
    return output


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    galaxy_id: int
    surface_id: int
    duration_s: float
    fluence_erg_kpc2: float
    peak_mach: float
    crossing_times_s: tuple[float, ...]
    attribution_status: str
    merger_event_ids: tuple[str, ...]


@dataclass(slots=True)
class ExposureInterval:
    start_s: float
    end_s: float
    galaxy_ids: np.ndarray
    records: tuple[ExposureRecord, ...]
    diagnostics: dict


def _intersection_interval(start, delta, normal, radius, half_width):
    """Clip a relative line segment to a finite circular shock-zone cylinder."""
    s0 = np.sum(start * normal, axis=-1)
    ds = np.sum(delta * normal, axis=-1)
    moving = ds != 0
    safe = np.where(moving, ds, 1.)
    root0, root1 = (-half_width - s0) / safe, (half_width - s0) / safe
    lo = np.where(moving, np.minimum(root0, root1), 0.)
    hi = np.where(moving, np.maximum(root0, root1), 1.)
    valid = moving | (np.abs(s0) <= half_width)
    tangent = start - s0[..., None] * normal
    velocity = delta - ds[..., None] * normal
    aa = np.sum(velocity * velocity, axis=-1)
    bb = 2 * np.sum(tangent * velocity, axis=-1)
    cc = np.sum(tangent * tangent, axis=-1) - radius**2
    disc = bb*bb - 4*aa*cc
    moving_tangent = aa > 0
    valid &= np.where(moving_tangent, disc >= 0, cc <= 0)
    denom = np.where(moving_tangent, 2*aa, 1.)
    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    radial_lo, radial_hi = (-bb - sqrt_disc) / denom, (-bb + sqrt_disc) / denom
    lo = np.maximum(np.maximum(lo, np.where(moving_tangent, radial_lo, 0)), 0)
    hi = np.minimum(np.minimum(hi, np.where(moving_tangent, radial_hi, 1)), 1)
    valid &= hi >= lo
    crossing = np.divide(-s0, ds, out=np.full_like(s0, np.nan), where=moving)
    # Half-open time convention counts a boundary crossing only once.
    crosses = valid & (crossing > 0) & (crossing <= 1) & (crossing >= lo) & (crossing <= hi)
    return lo, hi, valid, np.where(crosses, crossing, np.nan)


def _integrate_patch_union(segments):
    """Union duration and maximum linear flux, avoiding overlap double counts."""
    edges = {s[0] for s in segments} | {s[1] for s in segments}
    # Fluxes may exchange dominance inside an overlapping interval.
    for i, first in enumerate(segments):
        for second in segments[i+1:]:
            slope = first[3] - second[3]
            if slope != 0:
                t = (second[2] - first[2]) / slope
                if max(first[0], second[0]) < t < min(first[1], second[1]):
                    edges.add(t)
    duration = fluence = 0.
    edges = sorted(edges)
    for lo, hi in zip(edges, edges[1:]):
        mid = .5 * (lo + hi)
        active = [s for s in segments if s[0] <= mid <= s[1]]
        if not active:
            continue
        chosen = max(active, key=lambda s: s[2] + s[3]*mid)
        duration += hi-lo
        fluence += (hi-lo) * (chosen[2] + chosen[3]*mid)
    return duration, fluence


def integrate_galaxy_exposure(galaxy_ids, pos_previous_km, pos_current_km,
                              previous, current, *, attributions=None,
                              normal_cosine=.99, memory_budget_bytes=32*1024**2):
    """Integrate flux along galaxies relative to translating, matched patches.

    Galaxy positions have matching ID order. Geometry is interpolated linearly;
    flux is linear in time. Substantially rotating or unmatched patches are
    excluded and counted in diagnostics. Patch overlap on the same surface is
    combined using the maximum flux, while distinct surfaces remain separate.
    Fluence is an incident ICM exposure proxy, not energy absorbed by a galaxy.
    """
    ids = _ids(galaxy_ids, 'galaxy_ids')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('galaxy IDs must be unique')
    p0 = _array(pos_previous_km, (len(ids), 3), 'pos_previous_km')
    p1 = _array(pos_current_km, (len(ids), 3), 'pos_current_km')
    left, right, diagnostics = _matched(previous, current, normal_cosine)
    dt = current.time_s - previous.time_s
    normal = previous.normal[left] + current.normal[right]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    radii = np.minimum(previous.radius_km[left], current.radius_km[right])
    half = np.minimum(previous.half_width_km[left], current.half_width_km[right])
    grouped = {}
    for a, b in moving_patch_blocks(p0, p1, previous.pos_km[left], current.pos_km[right],
                                    radii, half, memory_budget_bytes=memory_budget_bytes):
        start = p0[a, None, :] - previous.pos_km[None, left[b], :]
        delta = ((p1[a]-p0[a])[:, None, :]
                 - (current.pos_km[right[b]]-previous.pos_km[left[b]])[None, :, :])
        lo, hi, valid, crossing = _intersection_interval(start, delta, normal[b], radii[b], half[b])
        for i, j in zip(*np.nonzero(valid)):
            l, r = left[b[j]], right[b[j]]
            key = (int(ids[a.start+i]), int(current.surface_id[r]))
            grouped.setdefault(key, []).append((float(lo[i, j]), float(hi[i, j]),
                float(previous.flux[l]), float(current.flux[r]-previous.flux[l]),
                float(max(previous.mach[l], current.mach[r])), float(crossing[i, j])))
    records = []
    attributions = {} if attributions is None else attributions
    for (galaxy, surface), segments in sorted(grouped.items()):
        duration, fluence = _integrate_patch_union(segments)
        fractions = sorted(s[5] for s in segments if np.isfinite(s[5]))
        distinct = []
        for fraction in fractions:
            if not distinct or fraction - distinct[-1] > 1.e-10:
                distinct.append(fraction)
        attribution = attributions.get(surface)
        if attribution is not None and attribution.surface_id != surface:
            raise ValueError('attribution surface IDs do not match mapping keys')
        records.append(ExposureRecord(galaxy, surface, duration*dt, fluence*dt,
            max(s[4] for s in segments), tuple(previous.time_s+t*dt for t in distinct),
            attribution.status if attribution else 'unattributed',
            attribution.event_ids if attribution else ()))
    diagnostics['fluence_unit'] = 'erg/kpc2'
    diagnostics['unobserved_patches_are_zero_exposure'] = False
    return ExposureInterval(previous.time_s, current.time_s, ids, tuple(records), diagnostics)


@dataclass(slots=True)
class ExposureAccumulator:
    """Accumulate non-overlapping intervals by persistent galaxy ID.

    Only summaries and last-covered times are retained. Durations sum across
    surfaces (surface-seconds); simultaneous distinct surfaces are additive.
    Missing intervals contribute to gap_s, not an assumption of no exposure.
    """
    summaries: dict = field(default_factory=dict)
    _last_end: dict = field(default_factory=dict)

    def add(self, interval):
        for galaxy in interval.galaxy_ids:
            if self._last_end.get(int(galaxy), -np.inf) > interval.start_s:
                raise ValueError('exposure intervals overlap or are out of order')
        for galaxy in interval.galaxy_ids:
            galaxy = int(galaxy)
            row = self.summaries.setdefault(galaxy, dict(duration_s=0., fluence_erg_kpc2=0.,
                candidate_merger_fluence_erg_kpc2=0., crossings=0, peak_mach=0., gap_s=0., covered_s=0., incomplete_intervals=0))
            if galaxy in self._last_end:
                row['gap_s'] += interval.start_s - self._last_end[galaxy]
            row['covered_s'] += interval.end_s - interval.start_s
            if (interval.diagnostics.get('unmatched_previous', 0)
                    or interval.diagnostics.get('unmatched_current', 0)
                    or interval.diagnostics.get('normal_rejected_patch_ids', [])):
                row['incomplete_intervals'] += 1
            self._last_end[galaxy] = interval.end_s
        for record in interval.records:
            row = self.summaries[record.galaxy_id]
            row['duration_s'] += record.duration_s
            row['fluence_erg_kpc2'] += record.fluence_erg_kpc2
            row['crossings'] += len(record.crossing_times_s)
            row['peak_mach'] = max(row['peak_mach'], record.peak_mach)
            if record.attribution_status == 'candidate':
                row['candidate_merger_fluence_erg_kpc2'] += record.fluence_erg_kpc2
