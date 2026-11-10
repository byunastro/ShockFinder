"""Local smooth backgrounds + mass surface + pressure/velocity shock strength."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import brentq
from collections import Counter
from functools import lru_cache
from . import _profile_geometry

NATIVE_MODEL = getattr(_profile_geometry, 'profile_model_kernel', None)

DIAGNOSTICS = Counter()


def fail(reason):
    DIAGNOSTICS[reason] += 1
    return None


GAMMA = 5 / 3
XG, WG = np.polynomial.legendre.leggauss(8)


@lru_cache(None)
def basis(n, degree):
    t = np.linspace(-1, 1, n)
    return np.linalg.pinv(np.column_stack([t**k for k in range(degree + 1)]))


@lru_cache(None)
def native_basis(n):
    table = np.zeros((4,n,n,3), order='F')
    for size in range(5,n+1):
        for degree in (1,2,3):
            table[:degree+1,:size,size-1,degree-1] = basis(size,degree)
    table.flags.writeable=False
    return table


def value(coeff, x):
    return coeff[0] + x * coeff[1] + x * x * coeff[2]


def density(coeff, x):
    return np.exp(np.clip(value(coeff, x), -700, 700))


def density_at(coeff, x):
    """Scalar evaluation with the same clamp and NaN behavior as density."""
    exponent = value(coeff, x)
    if exponent < -700:
        exponent = -700
    elif exponent > 700:
        exponent = 700
    return np.exp(exponent)


def integrate_density(coeff, a, b):
    x = (a + b) / 2 + (b - a) / 2 * XG
    return (b - a) / 2 * np.dot(WG, density(coeff, x))


def pressure_mach(ratio, gamma=GAMMA):
    return np.sqrt(np.maximum(((gamma - 1) + (gamma + 1) * ratio) / (2 * gamma), 1))


def upstream_state_errors(s, states, first, last, orders, center, upstream):
    """Bound upstream model sensitivity at the adopted surface, using fit data.

    Returns an error in log density and a relative pressure error. These are
    deterministic sensitivity bounds, not statistical confidence intervals.
    The alternative polynomial and the amplification of the measured fitting
    residual both matter: a small residual far from the surface is insufficient.
    No additional mesh samples or extra nonlinear fits are needed.
    """
    if hasattr(NATIVE_MODEL, 'upstream_state_errors'):
        return NATIVE_MODEL.upstream_state_errors(
            s, states, int(first)+1, int(last)+1, orders, center, upstream,
            native_basis(len(s)))
    x = s[first:last+1]
    scale = np.ptp(x)/2
    t = (x-x.mean())/scale
    target = (center-x.mean())/scale
    errors = np.empty(2)
    for var, degree in enumerate(orders):
        degree = int(degree)
        raw = states[first:last+1, var]
        if degree == 0:
            errors[var] = np.ptp(raw)/raw.mean()
            continue
        y = np.log(raw) if var == 0 else raw
        inverse = basis(len(x), degree)
        beta = inverse @ y
        higher = basis(len(x), degree+1) @ y
        powers = target**np.arange(degree+1)
        weights = powers @ inverse
        amplification = np.sum(abs(weights))
        residual = np.max(abs(y-sum(beta[k]*t**k for k in range(degree+1))))
        alternative = np.dot(higher, target**np.arange(degree+2))
        adopted = np.log(upstream[var]) if var == 0 else upstream[var]
        error = max(abs(alternative-adopted), residual*amplification)
        error += (32*np.finfo(float).eps * max(np.max(abs(y)), abs(adopted),
                  np.finfo(float).tiny) * max(amplification, 1.0))
        errors[var] = error if var == 0 else error/upstream[var]
    return errors


def reconstruct_batch(s, states, velocities, widths, normals, tolerance=0.01,
                      closure_tolerance=0.15, gamma=GAMMA):
    """Batch native work; preserve reference selection and difficult-profile fallback."""
    result = [None] * len(states)
    full = (np.isfinite(states).all(axis=(1,2)) & (states > 0).all(axis=(1,2))
            & np.isfinite(velocities).all(axis=1) & np.isfinite(widths).all(axis=1))
    if NATIVE_MODEL is None or not hasattr(NATIVE_MODEL, 'screen_batch'):
        full[:] = False
    for j in np.flatnonzero(~full):
        result[j] = reconstruct(s,states[j],velocities[j],widths[j],np.zeros(3),
                                normals[j],1.0,tolerance,closure_tolerance,gamma=gamma)
    rows = np.flatnonzero(full)
    if not len(rows):
        return result
    pressure = states[rows,:,1]
    derivative = np.gradient(gaussian_filter1d(np.log(pressure),1,axis=1,mode='nearest'),s,axis=1)
    pressure_gradient = np.gradient(pressure,s,axis=1)
    peaks = np.full(len(rows),-1,dtype=np.int32)
    slopes = np.zeros(len(rows))
    for j in range(len(rows)):
        candidates,_ = find_peaks(derivative[j],prominence=max(derivative[j].max()*1e-3,1e-12))
        if not len(candidates):
            fail('no_pressure_peak');continue
        local = candidates[abs(s[candidates])<=4]
        if not len(local):
            fail('no_local_pressure_peak');continue
        peak = local[np.argmax(derivative[j,local])]
        if derivative[j,peak]<=0:
            fail('negative_peak');continue
        peaks[j]=peak
        slopes[j]=pressure_gradient[j,abs(s-s[peak])<=2*max(1.0,widths[rows[j],peak])].max()
    use=np.flatnonzero(peaks>=0)
    if not len(use):
        return result
    selected=rows[use]
    native = NATIVE_MODEL.screen_batch(
        s,np.asfortranarray(states[selected].transpose(1,2,0)),
        np.asfortranarray(velocities[selected].T),np.asfortranarray(widths[selected].T),
        np.asfortranarray(derivative[use].T),peaks[use]+1,slopes[use],tolerance,1.0,
        gamma,closure_tolerance,native_basis(len(s)),XG,WG)
    for k,j in enumerate(use):
        item=tuple(a[...,k] for a in native)
        if not item[5].all():
            fail('no_resolved_background');continue
        if not item[7].any():
            fail('no_conservative_jump_model');continue
        original=int(rows[j])
        result[original]=reconstruct(s,states[original],velocities[original],widths[original],
            np.zeros(3),normals[original],1.0,tolerance,closure_tolerance,gamma=gamma,
            _prepared=(derivative[j],int(peaks[j]),float(slopes[j]),item))
    return result


def reconstruct(
    s,
    states,
    velocities,
    widths,
    anchor,
    normal,
    seed_width,
    tolerance=0.01,
    closure_tolerance=0.15,
    fixed_center=None,
    gamma=GAMMA,
    _prepared=None,
):
    good = (
        np.isfinite(states).all(axis=1) & np.isfinite(velocities) & np.isfinite(widths)
    )
    if not good.all():
        zero = np.argmin(abs(s))
        if not good[zero]:
            return fail("nonfinite_center")
        lo = zero
        hi = zero
        while lo > 0 and good[lo - 1]:
            lo -= 1
        while hi + 1 < len(good) and good[hi + 1]:
            hi += 1
        s = s[lo : hi + 1]
        states = states[lo : hi + 1]
        velocities = velocities[lo : hi + 1]
        widths = widths[lo : hi + 1]
        if len(s) < 5:
            return fail("short_profile")
    if states.min() <= 0:
        return fail("nonpositive")
    step = s[1] - s[0]
    rho, p = states.T
    if _prepared is None:
        # Reuse trapezoid terms, not cumulative sums: each candidate interval keeps
        # the original NumPy reduction order and therefore its rounding behavior.
        mass_terms = np.diff(s) * (rho[1:] + rho[:-1]) / 2.0
        derivative = np.gradient(gaussian_filter1d(np.log(p), 1, mode="nearest"), s)
        peaks, _ = find_peaks(
            derivative, prominence=max(derivative.max() * 1e-3, 1e-12 / seed_width)
        )
        if not len(peaks):
            return fail("no_pressure_peak")
        local_peaks = peaks[abs(s[peaks]) <= 4 * seed_width]
        if not len(local_peaks):
            return fail("no_local_pressure_peak")
        peak = local_peaks[np.argmax(derivative[local_peaks])]
        if derivative[peak] <= 0:
            return fail("negative_peak")
        local = abs(s - s[peak]) <= 2 * max(seed_width, widths[peak])
        pressure_slope = np.max(np.gradient(p, s)[local])
    else:
        derivative, peak, pressure_slope, native = _prepared
        mass_terms = None

    def physical(beta, center, scale):
        b = np.zeros(3)
        b[: len(beta)] = beta
        return np.array(
            [
                b[0] - center / scale * b[1] + center**2 / scale**2 * b[2],
                b[1] / scale - 2 * center / scale**2 * b[2],
                b[2] / scale**2,
            ]
        )

    def fit_background(ix):
        x = s[ix]
        center = x.mean()
        scale = (x[-1] - x[0]) / 2
        t = (x - center) / scale
        target = (s[peak] - center) / scale
        models = []
        errors = []
        orders = []
        for var in (0, 1):
            raw = states[ix, var]
            y = np.log(raw) if var == 0 else raw
            denominator = 1.0 if var == 0 else raw.mean()
            if np.ptp(raw) / raw.mean() <= tolerance:
                models.append(
                    np.array([np.log(raw.mean()) if var == 0 else raw.mean(), 0.0, 0.0])
                )
                errors.append(np.ptp(raw) / raw.mean())
                orders.append(0)
                continue
            choice = None
            for degree in (1, 2):
                beta = basis(len(x), degree) @ y
                pred = sum(beta[k] * t**k for k in range(degree + 1))
                res = np.max(abs(y - pred)) / denominator
                higher = basis(len(x), degree + 1) @ y
                at = sum(beta[k] * target**k for k in range(degree + 1))
                next_at = sum(higher[k] * target**k for k in range(degree + 2))
                uncertainty = abs(next_at - at) / denominator
                if res <= tolerance and (uncertainty <= tolerance or degree == 2):
                    choice = (
                        physical(beta, center, scale),
                        min(max(res, uncertainty), 1.0),
                        degree,
                    )
                    break
            if choice is None:
                return None
            coeff, error, order = choice
            models.append(coeff)
            errors.append(error)
            orders.append(order)
        pr = models[1]
        if pr[1] + 2 * s[peak] * pr[2] > 0.25 * max(pressure_slope, 1e-300):
            return None
        u = physical(basis(len(x), 2) @ velocities[ix], center, scale)
        return models, u, np.array(orders), float(np.linalg.norm(errors))

    def windows(side):
        candidates = range(peak - 1, 1, -1) if side < 0 else range(peak + 1, len(s) - 2)
        result = []
        last = None
        for edge in candidates:
            if last is not None and abs(s[edge] - last) < max(widths[edge], seed_width):
                continue
            n = max(5, int(np.ceil(2 * max(widths[edge], widths[peak]) / step)) + 1)
            ix = (
                np.arange(edge - n + 1, edge + 1)
                if side < 0
                else np.arange(edge, edge + n)
            )
            if ix[0] < 0 or ix[-1] >= len(s):
                continue
            if s[ix[-1]] - s[ix[0]] + step / 2 < 2 * widths[ix].max():
                continue
            if len(result) >= 4 and np.any(
                np.ptp(states[ix], axis=0) / states[ix].mean(axis=0) > tolerance
            ):
                continue
            fit = fit_background(ix)
            if fit is not None:
                result.append((ix, *fit))
                last = s[edge]
                if sum(not r[3].any() for r in result) >= 2:
                    break
        # Keep the nearest smooth models and the nearest resolved constant state.
        flat = [r for r in result if not r[3].any()]
        chosen = result[:4]
        for r in flat[:2]:
            if all(r is not q for q in chosen):
                chosen.append(r)
        return chosen

    if NATIVE_MODEL is not None:
        def native_windows(side):
            if _prepared is None:
                coeff,vel,orders,errors,bounds,count = NATIVE_MODEL.fit_windows(
                    s, states, velocities, widths, int(peak)+1, side, pressure_slope,
                    tolerance, seed_width, native_basis(len(s)))
            else:
                k = 0 if side < 0 else 1
                coeff,vel,orders,errors,bounds = [a[...,k] for a in native[:5]]
                count = native[5][k]
            return [(np.arange(bounds[0,k]-1,bounds[1,k]),
                     [coeff[:,0,k],coeff[:,1,k]],vel[:,k],orders[:,k],float(errors[k]))
                    for k in range(count)]
        left,right = native_windows(-1),native_windows(1)
    else:
        left, right = windows(-1), windows(1)
    if not left or not right:
        return fail("no_resolved_background")
    left = [(*f,i) for i,f in enumerate(left)]
    right = [(*f,i) for i,f in enumerate(right)]
    candidates = []
    flat_pairs, other_pairs = [], []
    left_flat = [not r[3].any() for r in left]
    right_flat = [not r[3].any() for r in right]
    for i, l in enumerate(left):
        for j, r in enumerate(right):
            (flat_pairs if left_flat[i] and right_flat[j] else other_pairs).append((l, r))
    # The original selection always prefers valid constant/constant models.
    # Evaluate those first; discarded nonconstant models cannot affect the result.
    for pair_index, ((il, cl, ul, ol, errl, li), (ir, cr, ur, orr, errr, ri)) in enumerate(
        flat_pairs + other_pairs
    ):
        if pair_index == len(flat_pairs) and candidates:
            break
        if _prepared is not None and not native[7][li,ri]:
            continue
        # A weak secondary ripple must not borrow the jump of a different,
        # stronger pressure peak included in its background-fit interval.
        if derivative[il[0] : ir[-1] + 1].max() > derivative[peak] * (1 + 1e-10):
            continue
        a = s[il[-1]]
        b = s[ir[0]]
        integral = mass_terms[il[-1] : ir[0]].sum() if _prepared is None else 0.0

        def mass(c):
            return (
                integrate_density(cl[0], a, c)
                + integrate_density(cr[0], c, b)
                - integral
            )

        if _prepared is not None:
            center = native[6][li,ri]
        elif NATIVE_MODEL is not None:
            center,status = NATIVE_MODEL.mass_root(cl[0],cr[0],a,b,integral,seed_width,XG,WG)
            if status:
                continue
        else:
            fl, fh = mass(a), mass(b)
            if not np.isfinite([fl, fh]).all() or np.signbit(fl) == np.signbit(fh):
                continue
            try:
                center = brentq(mass, a, b, xtol=1e-10 * seed_width, rtol=1e-12, maxiter=60)
            except (ValueError, RuntimeError):
                continue
        free_center = center
        if fixed_center is not None:
            if abs(center - fixed_center) > 0.25 * seed_width:
                continue
            center = float(fixed_center)
        if abs(center - s[peak]) > 2 * max(widths[peak], seed_width):
            continue
        up = np.array([density_at(cl[0], center), value(cl[1], center)])
        down = np.array([density_at(cr[0], center), value(cr[1], center)])
        if np.min([*up, *down]) <= 0:
            continue
        support_error = upstream_state_errors(
            s, states, int(il[0]), int(il[-1]), ol, center, up)
        if not np.isfinite(support_error).all() or np.any(support_error > tolerance):
            continue
        r = down[0] / up[0]
        rp = down[1] / up[1]
        if r <= 1 or rp <= 1:
            continue
        entropy = rp / r**gamma
        # Positive entropy production remains a physical requirement. Background
        # model sensitivity is NOT a calibrated noise distribution: report a
        # weak entropy signal as uncertain rather than silently censoring it.
        if np.log(entropy) <= 1e-10:
            continue
        m = pressure_mach(rp, gamma)
        re = (gamma + 1) * m * m / ((gamma - 1) * m * m + 2)
        sound = np.sqrt(gamma * up[1] / up[0])
        jump = m * sound * (1 - 1 / re)
        vl = value(ul, center)
        vr = value(ur, center)
        dr = abs(r - re) / max(abs(r - 1), 1e-12)
        dv = abs((vl - vr) - jump) / max(abs(jump), 1e-12 * sound)
        # Density is a QUALITY SCORE, not a hard veto on unresolved Sedov peaks.
        if vl <= vr or dv > closure_tolerance:
            continue
        quality = dr * dr + dv * dv + errl * errl + errr * errr
        candidates.append(
            dict(
                mach=float(m),
                offset=float(center),
                mass_surface_displacement=float(free_center - center),
                center_constraint_active=fixed_center is not None,
                peak_offset=float(s[peak]),
                density_residual=float(dr),
                velocity_residual=float(dv),
                upstream_state_error=support_error,
                entropy_ratio=float(entropy),
                entropy_margin=float(np.log(entropy) / max((1+gamma)*(errl+errr),1e-30)),
                normal_velocity=np.array([vl,vr]),
                background_error=np.array([errl,errr]),
                background_indices=np.array([[il[0],il[-1]],[ir[0],ir[-1]]]),
                background_bounds=np.array([[s[il[0]],s[il[-1]]],[s[ir[0]],s[ir[-1]]]]),
                upstream=up,
                downstream=down,
                window=np.array([a, b]),
                point=anchor + center * normal,
                orders=np.r_[ol, orr],
                quality=float(quality),
            )
        )
    if not candidates:
        return fail("no_conservative_jump_model")
    flat = [r for r in candidates if not r["orders"].any()]
    family = flat or candidates
    best = min(family, key=lambda r: r["quality"])
    # Ranges are conditional on the resolved, physically admissible background
    # family. They are sensitivity diagnostics, NEVER statistical intervals.
    best['model_count'] = len(family)
    best['mach_range'] = np.array([min(f['mach'] for f in family),max(f['mach'] for f in family)])
    best['offset_range'] = np.array([min(f['offset'] for f in family),max(f['offset'] for f in family)])
    up,down=best['upstream'],best['downstream']
    contrast=down[0]-up[0]
    best['center_sensitivity'] = float(np.diff(best['window'])[0] *
        np.dot(best['background_error'],[up[0],down[0]])/contrast)
    vl,vr=best['normal_velocity']
    vs=(down[0]*vr-up[0]*vl)/contrast
    w1,w2=vl-vs,vr-vs
    best['shock_speed_mass']=float(vs)
    best['shock_speed_pressure']=float(vl-best['mach']*np.sqrt(gamma*up[1]/up[0]))
    momentum1=up[1]+up[0]*w1*w1
    momentum2=down[1]+down[0]*w2*w2
    best['momentum_residual']=float(abs(momentum2-momentum1)/
        max(abs(down[1]-up[1]),1e-30))
    h1=gamma/(gamma-1)*up[1]/up[0]
    h2=gamma/(gamma-1)*down[1]/down[0]
    kinetic=.5*(w1*w1-w2*w2)
    best['energy_residual']=float(abs(h1-h2+kinetic)/max(abs(h1-h2),abs(kinetic),gamma*up[1]/up[0],1e-30))
    return best
