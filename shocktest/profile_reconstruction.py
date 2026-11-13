"""The single production shock estimator: local profile reconstruction."""

from collections import Counter
import time
import numpy as np
from .profile_geometry import make_geometry
from ._profile_model import reconstruct
from ._profile_model import basis
from .profile_quality import quality_flags
from . import _profile_model as profile_model

K_B = 1.380649e-16
M_P = 1.67262192369e-24
MSUN = 1.98847e33
KPC = 3.0856775814913673e21


def normalize(a):
    length = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.divide(a, length, out=np.zeros_like(a), where=length > 0)


def density_cgs(unit):
    factors = {
        "Msol/kpc3": MSUN / KPC**3,
        "g/cc": 1.0,
        "g/cm3": 1.0,
        "g/cm^3": 1.0,
        "kg/m3": 1e-3,
    }
    if unit not in factors:
        raise ValueError(
            f"Unsupported density unit {unit!r} for thermal pressure conversion"
        )
    return factors[unit]


def prepare(arrays, finder, pressure=None):
    """Prepare one global mesh; partitioning must never change normalization."""
    p = arrays["pos"]
    dx = arrays["dx"]
    rho = arrays["rho"]
    temperature = arrays["temp"]
    if finder.temperature_unit != "K":
        raise ValueError('Profile reconstruction requires temperature_unit="K"')
    velocity_factor = {"km/s": 1.0, "m/s": 1e-3, "cm/s": 1e-5}.get(finder.velocity_unit)
    if velocity_factor is None:
        raise ValueError("velocity_unit must be km/s, m/s, or cm/s")
    qfactor = K_B / (finder.mean_molecular_weight * M_P) / 1e10
    q = temperature * qfactor
    if pressure is not None:
        unit = finder.thermal_pressure_unit
        factors = {"erg/cm3": 1.0, "erg/cm^3": 1.0, "dyn/cm2": 1.0, "Pa": 10.0}
        if unit not in factors:
            raise ValueError("Set thermal_pressure_unit to erg/cm3, dyn/cm2, or Pa")
        q = pressure * factors[unit] / (rho * density_cgs(finder.density_unit)) / 1e10
    if (
        np.any(rho <= 0)
        or np.any(temperature <= 0)
        or np.any(q <= 0)
        or not np.isfinite(q).all()
    ):
        raise ValueError(
            "Profile reconstruction requires positive finite density, temperature and thermal pressure"
        )
    density_scale = float(np.median(rho))
    speed_scale = float(np.sqrt(np.median(q)))
    r = rho / density_scale
    pr = r * q / speed_scale**2
    v = arrays["vel"] * velocity_factor / speed_scale
    geometry = make_geometry(p, dx)
    eligible=np.flatnonzero(arrays['detection_mask'] != 0)
    # Seed cuts limit work, never the sampling mesh. Normals at newly located
    # center leaves are evaluated lazily with exactly the same face stencil.
    div, grad, gradp = geometry.derivatives_at(temperature, v, pr, eligible)
    normals=np.zeros((len(p),3));known=np.zeros(len(p),bool)
    normals[eligible]=normalize(grad);known[eligible]=True
    mask = (
        (div < 0)
        & (np.linalg.norm(grad, axis=1) > 0)
        & (np.sum(gradp * grad, axis=1) > 0)
    )
    seeds = eligible[mask]
    diagnostics = Counter(
        seed_cells=len(seeds), geometry_native=int(getattr(geometry, "native", False))
    )
    diagnostics['geometry_cache_native'] = int(
        hasattr(getattr(geometry, 'kernel', None), 'sample_batch_cached'))
    # All components are volume-averaged together. Momentum is divided by the
    # sampled density only AFTER conservative restriction/CIC and tube averaging.
    values = np.asfortranarray(np.column_stack((r, pr, r[:, None] * v)))
    return dict(pos=geometry.pos, dx=geometry.dx, temperature=temperature,
                velocity=np.asfortranarray(v), pressure=pr, normals=normals,
                known=known, values=values, seeds=seeds, geometry=geometry,
                density_scale=density_scale, speed_scale=speed_scale,
                qfactor=qfactor, diagnostics=diagnostics)


def fit_precedes(fit, old):
    """Preserve serial quality and traversal ties after arbitrary task ordering."""
    return (not fit.get('upstream_reliable', True), fit['quality'], fit['_order']) < (
        not old.get('upstream_reliable', True), old['quality'], old['_order'])


def scan(context, finder, batch_starts=None, *, best=None, normal_cache=None):
    """Scan original numerical batches against the complete immutable mesh.

    Workers own their sparse result and lazy-normal dictionaries. No shared
    array is written, including normals of centers outside the seed selection.
    """
    p, dx = context['pos'], context['dx']
    geometry, seeds = context['geometry'], context['seeds']
    temperature, v, pr = context['temperature'], context['velocity'], context['pressure']
    normals, known, values = context['normals'], context['known'], context['values']
    density_scale, speed_scale, qfactor = (context[k] for k in
        ('density_scale', 'speed_scale', 'qfactor'))
    normal_cache = {} if normal_cache is None else normal_cache
    def normal_at(rows):
        rows = np.asarray(rows, dtype=np.int64)
        missing = np.array([int(i) for i in np.unique(rows[~known[rows]])
                            if int(i) not in normal_cache], dtype=np.int64)
        if len(missing):
            _, g, _ = geometry.derivatives_at(temperature, v, pr, missing)
            normal_cache.update(zip(map(int, missing), normalize(g)))
        out = normals[rows].copy()
        for j in np.flatnonzero(~known[rows]):
            out[j] = normal_cache[int(rows[j])]
        return out

    offsets = np.arange(-12, 12.001, 0.25)
    # Keep only the current winner for each output leaf, rather than retaining
    # every converged seed model until the entire snapshot has been scanned.
    best = {} if best is None else best
    diagnostics = Counter()
    accepted_models = 0

    def profiles(ids, anchors, directions):
        h = dx[ids].copy()
        for off in (-12, -8, -4, 0, 4, 8, 12):
            near = geometry.locate(anchors + off * dx[ids, None] * directions)
            ok = near >= 0
            h[ok] = np.maximum(h[ok], dx[near[ok]])
        basis = np.eye(3)[np.argmin(abs(directions), axis=1)]
        t1 = normalize(np.cross(directions, basis))
        t2 = np.cross(directions, t1)
        tube = np.stack(
            (
                np.zeros_like(t1),
                0.5 * dx[ids, None] * t1,
                -0.5 * dx[ids, None] * t1,
                0.5 * dx[ids, None] * t2,
                -0.5 * dx[ids, None] * t2,
            ),
            axis=1,
        )
        points = (
            anchors[:, None, None, :]
            + h[:, None, None, None]
            * offsets[None, :, None, None]
            * directions[:, None, None, :]
            + tube[:, None, :, :]
        )
        widths = np.broadcast_to(h[:, None, None], points.shape[:-1]).ravel()
        sampled = geometry.sample(points.reshape(-1, 3), widths, values).reshape(
            len(ids), len(offsets), 5, 5
        )
        complete = np.isfinite(sampled).all(axis=(2, 3))
        average = np.einsum("k,ijkv->ijv", [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], sampled)
        # At an extracted region's transverse boundary, use the supported
        # central ray instead of inventing gas outside the input volume.
        average[~complete] = sampled[:, :, 0, :][~complete]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (
                np.sum(average[:, :, 2:] * directions[:, None, :], axis=2)
                / average[:, :, 0]
            )
        containing = geometry.locate(points.reshape(-1, 3)).reshape(
            len(ids), len(offsets), 5
        )
        covered = np.where(containing >= 0, dx[np.maximum(containing, 0)], 0).max(
            axis=2
        )
        covered = np.maximum(covered, h[:, None]) / h[:, None]
        vector_velocity = average[:, :, 2:] / average[:, :, :1]
        return h, average[:, :, :2], u, covered, vector_velocity, complete

    batch = int(finder.profile_batch_size)
    progress_step = (int(finder.progress_interval) if finder.progress_interval and finder.progress_interval > 0
                     else max(1, (len(seeds) + 19) // 20))
    next_progress = progress_step
    processed = 0
    progress_start = last_progress = time.monotonic()
    if batch_starts is None:
        batch_starts = range(0, len(seeds), batch)
    for start in batch_starts:
        start = int(start)
        if start < 0 or start >= len(seeds) or start % batch:
            raise ValueError('Task offsets must identify original complete seed batches')
        ids = seeds[start : start + batch]
        processed += len(ids)
        anchors = p[ids].copy()
        direction = normal_at(ids)
        initial = ids.copy()
        for iteration in range(6):
            if not len(ids):
                break
            h, states, u, covering, vector_velocity, complete = profiles(ids, anchors, direction)
            batch_fits = profile_model.reconstruct_batch(
                offsets, states, u, covering, direction, finder.sampling_tolerance,
                0.15, gamma=finder.gamma)
            again = []
            for j in range(len(ids)):
                fit = batch_fits[j]
                if fit is None:
                    diagnostics["profile_rejected"] += 1
                    continue
                point = anchors[j] + fit["offset"] * h[j] * direction[j]
                cell = geometry.locate(point[None, :])[0]
                final_normal = normal_at([cell])[0] if cell >= 0 else np.zeros(3)
                if cell < 0 or not np.any(final_normal):
                    diagnostics["center_outside_or_undefined"] += 1
                    continue
                consistent = np.allclose(
                    final_normal, direction[j], rtol=1e-12, atol=1e-12
                )
                if iteration == 5 and not consistent:
                    fit = reconstruct(
                        offsets,
                        states[j],
                        u[j],
                        covering[j],
                        np.zeros(3),
                        direction[j],
                        1.0,
                        finder.sampling_tolerance,
                        0.15,
                        fixed_center=0.0,
                        gamma=finder.gamma,
                    )
                    if fit is None:
                        diagnostics["normal_nonconvergence"] += 1
                        continue
                    point = anchors[j]
                    cell = int(ids[j])
                    final_normal = normal_at([cell])[0]
                    consistent = True
                if iteration >= 1 and consistent:
                    if fit["mach"] <= finder.min_mach:
                        diagnostics["mach_rejected"] += 1
                        continue
                    np.testing.assert_allclose(
                        final_normal, direction[j], rtol=1e-12, atol=1e-12
                    )
                    support = anchors[j] + fit["window"][:, None] * h[j] * direction[j]
                    endpoints = geometry.locate(support)
                    if np.any(endpoints < 0):
                        diagnostics["support_missing"] += 1
                        continue
                    accepted_models += 1
                    fit['_order'] = (start, iteration, int(initial[j]))
                    if int(cell) in best and not fit_precedes(fit, best[int(cell)]):
                        continue
                    fit.update(
                        cell=int(cell),
                        point=point.copy(),
                        normal=final_normal.copy(),
                        seed=int(initial[j]),
                        passes=iteration + 1,
                        reference_width=float(h[j]),
                        support_pos=support,
                        endpoints=endpoints,
                        density=fit["upstream"][0] * density_scale,
                        temperature=fit["upstream"][1]
                        / fit["upstream"][0]
                        * speed_scale**2
                        / qfactor,
                    )
                    states_out = np.stack((fit["upstream"], fit["downstream"]))
                    fit["states_density"] = states_out[:, 0] * density_scale
                    fit["states_temperature"] = (
                        states_out[:, 1] / states_out[:, 0] * speed_scale**2 / qfactor
                    )
                    fit["states_pressure"] = (
                        fit["states_density"]
                        * density_cgs(finder.density_unit)
                        * fit["states_temperature"]
                        * K_B
                        / (finder.mean_molecular_weight * M_P)
                    )
                    # The same quadratic velocity backgrounds used for the RH
                    # normal jump, now retained as full vectors for galaxy work.
                    velocities_out=[]
                    for lo,hi in fit['background_bounds']:
                        ix=np.flatnonzero((offsets>=lo-1e-12)&(offsets<=hi+1e-12))
                        ctr=offsets[ix].mean();scale=(offsets[ix[-1]]-offsets[ix[0]])/2
                        beta=basis(len(ix),2) @ vector_velocity[j,ix]
                        target=(fit['offset']-ctr)/scale
                        velocities_out.append(np.array([1,target,target*target]) @ beta)
                    fit['states_velocity']=np.array(velocities_out)*speed_scale
                    active=(offsets>=fit['background_bounds'][0,0])&(offsets<=fit['background_bounds'][1,1])
                    fit['quality_flags']=quality_flags(fit,bool(complete[j,active].all()))
                    fit['shock_speed_mass']*=speed_scale
                    fit['shock_speed_pressure']*=speed_scale
                    best[int(cell)] = fit
                else:
                    again.append((cell, point, initial[j]))
            ids = np.array([x[0] for x in again], dtype=np.int64)
            anchors = np.array([x[1] for x in again]).reshape(-1, 3)
            initial = np.array([x[2] for x in again], dtype=np.int64)
            direction = normal_at(ids)
        if finder.show_progress:
            done = processed
            now = time.monotonic()
            if done >= next_progress or done == len(seeds) or now - last_progress >= 60:
                elapsed = max(now - progress_start, 1e-12)
                finder._progress(
                    f"Profile reconstruction: {done:,}/{len(seeds):,} seeds "
                    f"({done/elapsed:.1f} seeds/s; {len(best):,} unique centers)"
                )
                next_progress = done + progress_step
                last_progress = now
    diagnostics['processed_seeds'] = processed
    diagnostics['accepted_models'] = accepted_models
    return best, diagnostics, normal_cache


def finish(context, best, counts, derivative_cells):
    diagnostics = Counter(context['diagnostics'])
    diagnostics.update(counts)
    diagnostics["duplicate_leaf_models"] = diagnostics.pop('accepted_models', 0) - len(best)
    diagnostics["accepted_centers"] = len(best)
    diagnostics['quality_clear_centers']=sum(f['quality_flags']==0 for f in best.values())
    diagnostics['upstream_sensitive_centers']=sum(not f['upstream_reliable'] for f in best.values())
    diagnostics['upstream_alternative_centers']=sum(f['upstream_alternative'] for f in best.values())
    diagnostics['model_native']=int(profile_model.NATIVE_MODEL is not None)
    diagnostics['model_batch_native']=int(hasattr(profile_model.NATIVE_MODEL,'screen_batch'))
    diagnostics['derivative_cells'] = int(derivative_cells)
    for fit in best.values():
        fit.pop('_order', None)
    return [best[k] for k in sorted(best)], dict(diagnostics)


def detect(arrays, finder, pressure=None):
    """The fixed production estimator; every selected candidate is scanned."""
    context = prepare(arrays, finder, pressure)
    best, counts, lazy = scan(context, finder)
    return finish(context, best, counts, context['known'].sum() + len(lazy))
