"""Shared selection for plotted and analyzed shock samples.

No secondary Mach estimators or persistent full-mesh validity arrays are used.
Profile residuals measure ideal-gas jump consistency, not shock origin or truth.
"""

import copy
import numpy as np


def valid_shock_rows(result, *, min_mach=1.0, closure_tolerance=0.15,
                     closure_max_mach=3.0, require_upstream_reliable=False):
    """Return selected retained-row indices, in ascending order.

    Profile results require a resolved positive entropy signal (entropy_margin
    >1), finite nonnegative jump residuals and velocity residual <=0.15.
    Density/momentum/energy residuals must also be <=0.15 for Mach <=3;
    unresolved strong-shock downstream peaks can fail these diagnostics even
    with an accurate pressure Mach. Set closure_max_mach=None to enforce them
    at every Mach. Saved upstream-state errors must be finite and nonnegative;
    relative pressure sensitivity must be <1 (no pressure cancellation).
    A 1% upstream precision target is not a default shock-existence veto.
    require_upstream_reliable=True additionally applies the stored fitting
    tolerance to both upstream sensitivities for an explicit quality subset.
    Earlier
    profile archives without those errors receive the residual checks only;
    their upstream extrapolation cannot be validated retroactively here.

    Historical temperature-jump results use saved mach_consistent/status when
    present. Other manually constructed results retain their shock mask.
    Diagnostic flags for center/model sensitivity, bounded normals,
    incomplete tubes and single backgrounds are not blanket rejection rules.
    These are result rows, not input IDs or slots in result.sampling.
    """
    mach = np.asarray(result.mach)
    shock = np.asarray(result.shock)
    if mach.ndim != 1 or shock.shape != mach.shape or shock.dtype.kind != 'b':
        raise ValueError('mach and boolean shock must be aligned one-dimensional arrays')
    if not np.isfinite(min_mach):
        raise ValueError('min_mach must be finite')
    if not isinstance(require_upstream_reliable, (bool, np.bool_)):
        raise ValueError('require_upstream_reliable must be boolean')
    if not np.isfinite(closure_tolerance) or closure_tolerance < 0:
        raise ValueError('closure_tolerance must be nonnegative and finite')
    if closure_max_mach is not None and (not np.isfinite(closure_max_mach) or closure_max_mach < 1):
        raise ValueError('closure_max_mach must be None or a finite Mach >=1')
    if getattr(result, 'mach_estimator', None) == 'profile_pressure':
        sampling = getattr(result, 'sampling', None)
        if not isinstance(sampling, dict) or 'rows' not in sampling:
            raise ValueError('Profile selection requires saved sampling diagnostics')
        rows = np.asarray(sampling['rows'])
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or
                (len(rows) and (rows[0] < 0 or rows[-1] >= len(mach))) or
                np.any(rows[1:] <= rows[:-1])):
            raise ValueError('sampling rows must be sorted unique retained-row indices')
        keep = shock[rows] & np.isfinite(mach[rows]) & (mach[rows] >= min_mach)
        weak = (np.ones(len(rows), dtype=bool) if closure_max_mach is None
                else mach[rows] <= closure_max_mach)
        if 'entropy_margin' not in sampling:
            raise ValueError('Profile selection requires sampling[\'entropy_margin\']')
        entropy = np.asarray(sampling['entropy_margin'])
        if entropy.shape != rows.shape:
            raise ValueError('entropy_margin must align with sampling rows')
        keep &= np.isfinite(entropy) & (entropy > 1)
        for name in ('density_residual', 'velocity_residual',
                     'momentum_residual', 'energy_residual'):
            if name not in sampling:
                raise ValueError(f'Profile selection requires sampling[{name!r}]')
            residual = np.asarray(sampling[name])
            if residual.shape != rows.shape:
                raise ValueError(f'sampling[{name!r}] must align with sampling rows')
            keep &= np.isfinite(residual) & (residual >= 0)
            keep &= ((residual <= closure_tolerance) if name == 'velocity_residual'
                     else (~weak | (residual <= closure_tolerance)))
        if 'upstream_state_error' in sampling:
            error = np.asarray(sampling['upstream_state_error'])
            if error.shape != (len(rows), 2):
                raise ValueError('upstream_state_error must have shape (K, 2)')
            keep &= (np.isfinite(error).all(axis=1) & (error >= 0).all(axis=1)
                     & (error[:, 1] < 1))
            if require_upstream_reliable:
                tolerance = float(getattr(result, 'sampling_tolerance', 0.01))
                if not np.isfinite(tolerance) or tolerance < 0:
                    raise ValueError('sampling_tolerance must be nonnegative and finite')
                keep &= (error <= tolerance).all(axis=1)
        elif require_upstream_reliable:
            raise ValueError('Upstream quality selection requires saved upstream_state_error')
        return rows[keep]

    if require_upstream_reliable:
        raise ValueError('Upstream quality selection requires profile sampling diagnostics')
    # Restrict work to detected rows; never expand historical status into six
    # additional full-mesh boolean arrays merely to select a plot.
    rows = np.flatnonzero(shock)
    keep = np.isfinite(mach[rows]) & (mach[rows] >= min_mach)
    consistent = getattr(result, 'mach_consistent', None)
    if consistent is not None:
        consistent = np.asarray(consistent)
        if consistent.shape != mach.shape or consistent.dtype.kind != 'b':
            raise ValueError('mach_consistent must be a boolean array aligned with mach')
        keep &= consistent[rows]
    else:
        status = getattr(result, 'mach_validation_status', None)
        if status is not None:
            status = np.asarray(status)
            if status.shape != mach.shape or status.dtype.kind not in 'iu':
                raise ValueError('mach_validation_status must be an integer array aligned with mach')
            keep &= (status[rows] & 128) != 0
    return rows[keep]


def shock_valid_mask(result, *, min_mach=1.0, closure_tolerance=0.15,
                     closure_max_mach=3.0, require_upstream_reliable=False):
    """Dense equivalent of valid_shock_rows; allocates one bool per result row.

    Prefer valid_shock_rows for large datasets when a dense mask is unnecessary.
    """
    rows = valid_shock_rows(result, min_mach=min_mach, closure_tolerance=closure_tolerance,
                           closure_max_mach=closure_max_mach,
                           require_upstream_reliable=require_upstream_reliable)
    mask = np.zeros(np.shape(result.mach), dtype=bool)
    mask[rows] = True
    return mask


def filter_shocks(result, *, min_mach=1.0, closure_tolerance=0.15,
                  closure_max_mach=3.0, require_upstream_reliable=False):
    """Create a selected result for catalog/analysis without rerunning detection.

    Dense fields are shared; a new shock mask and selected sparse diagnostics
    are allocated. Do not edit shared dense fields in-place. All row indices
    remain in the original retained mesh. Rebuild catalogs/labels from this
    result: an existing catalog summarizes the original, unfiltered sample.
    """
    selected = copy.copy(result)
    selected.shock = shock_valid_mask(result, min_mach=min_mach,
                                     closure_tolerance=closure_tolerance,
                                     closure_max_mach=closure_max_mach,
                                     require_upstream_reliable=require_upstream_reliable)
    sampling = getattr(result, 'sampling', None)
    if sampling is not None:
        rows = np.asarray(sampling['rows'])
        keep = selected.shock[rows]
        selected.sampling = {
            key: (value[keep].copy() if isinstance(value, np.ndarray)
                  and value.ndim and len(value) == len(rows) else value)
            for key, value in sampling.items()
        }
    if getattr(result, 'diagnostics', None) is not None:
        selected.diagnostics = dict(result.diagnostics)
        selected.diagnostics['selected_centers'] = int(np.count_nonzero(selected.shock))
    return selected
