"""Notebook plotting and safe NPZ exports for profile reconstruction results."""

from pathlib import Path
import json, numpy as np
from .profile_quality import ProfileQualityFlag


def save_profiles(path, result, metadata=None):
    rows = np.flatnonzero(result.shock)
    s = result.sampling
    data = dict(
        mach=result.mach[rows],
        position=result.center_position[rows],
        cell_position=result.pos[rows],
        normal=result.normal[rows],
        width=result.dx[rows],
        input_row=result.selected_indices[rows],
    )
    for name in (
        "temperature",
        "density",
        "pressure",
        "support_pos",
        "reference_width",
        "density_residual",
        "velocity_residual",
        "center_constraint_active",
        "mass_surface_displacement",
        "profile_passes",
        "quality",
        "background_orders",
    ):
        data[name] = s[name]
    for name in ('upstream_state_error','upstream_reliable','upstream_alternative',
                 'downstream_state_error','downstream_reliable','downstream_alternative',
                 'downstream_recovered','sampling_origin','sampling_cell_width','background_window_bounds',
                 'pressure_ratio_range','mach_sensitivity_range','pressure_jump_supported',
                 'flow_mach','rh_compatible','admissible_mach_range','admissible_model_count',
                 'velocity','entropy_margin','center_sensitivity','momentum_residual','energy_residual',
                 'shock_speed_mass','shock_speed_pressure','mach_model_range','center_model_range',
                 'background_model_count','profile_quality_flags','quality_clear'):
        if name in s:data[name]=s[name]
    data["metadata_json"] = json.dumps(
        dict(
            metadata or {},
            mach_estimator=result.mach_estimator,
            position_unit=result.position_unit,
            density_unit=result.density_unit,
            pressure_unit=result.pressure_unit,
            velocity_unit='km/s',
            gamma=result.gamma,
            mean_molecular_weight=result.mean_molecular_weight,
            profile_quality_definitions={flag.name:int(flag) for flag in ProfileQualityFlag},
            diagnostics=result.diagnostics,
            measurement_info=getattr(result, 'measurement_info', None),
        ),
        default=lambda a: a.tolist() if isinstance(a, np.ndarray) else a.item(),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def save_slice(path, cell, *, axis='z', coordinate=None, metadata=None):
    """Export a thin leaf slice for notebook plotting WITHOUT rerunning detection.

    The reader's physical units are required: km, K, g/cc. Only intersecting
    leaves are copied; no full-mesh position temporary is allocated.
    """
    if axis not in 'xyz' or len(axis)!=1:raise ValueError('axis must be x, y or z')
    q=np.asarray(cell[axis,'km']);dx=np.asarray(cell['dx','km'])
    plane=float(np.median(q) if coordinate is None else coordinate)
    rows=np.flatnonzero((q-dx/2<=plane)&(q+dx/2>plane))
    if not len(rows):raise ValueError('Slice plane does not intersect any input leaf')
    meta=dict(metadata or {},schema='shockfinder-slice-v1',axis=axis,coordinate=plane,
        position_unit='km',density_unit='g/cc',temperature_unit='K')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,position=np.column_stack([cell[a,'km'][rows] for a in 'xyz']),
        width=dx[rows],temperature=cell['T','K'][rows],density=cell['rho','g/cc'][rows],
        metadata_json=json.dumps(meta,default=lambda x:np.asarray(x).tolist()))
    return path


def plot_saved_slice(path, *, field='density', profiles=None, log=True):
    """Plot exported slice/profiles NPZ files; no raw snapshot or finder needed."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    if field not in ('density','temperature'):raise ValueError('field must be density or temperature')
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['metadata_json']));axis='xyz'.index(meta['axis'])
        dims=[i for i in range(3) if i!=axis];unit=3.0856775814913673e16
        p=a['position'][:,dims]/unit;half=a['width']/unit/2
        vertices=p[:,None,:]+half[:,None,None]*np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        values=a[field];values=np.log10(np.maximum(values,np.finfo(float).tiny)) if log else values
        fig,ax=plt.subplots(figsize=(6,5));poly=PolyCollection(vertices,array=values,cmap='magma',edgecolors='none',rasterized=True)
        ax.add_collection(poly);ax.autoscale_view();ax.set_aspect('equal')
        label=field+' ['+meta[field+'_unit']+']'
        fig.colorbar(poly,ax=ax,label=('log10 ' if log else '')+label)
        ax.set(xlabel='xyz'[dims[0]]+' [kpc]',ylabel='xyz'[dims[1]]+' [kpc]')
        if profiles is not None:
            with np.load(profiles,allow_pickle=False) as b:
                q=b['cell_position'][:,axis];h=b['width'];plane=meta['coordinate']
                keep=(q-h/2<=plane)&(q+h/2>plane)
                ax.scatter(b['position'][keep][:,dims[0]]/unit,b['position'][keep][:,dims[1]]/unit,
                           s=3,color='cyan',label='Profile centers');ax.legend()
        fig.tight_layout();return fig,ax


def plot_slice(
    cell, *, field=("rho", "g/cc"), axis="z", coordinate=None, result=None, log=True
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from .pyShockFinder import KPC

    xyz = np.column_stack([cell[a, "km"] for a in "xyz"])
    dx = cell["dx", "km"]
    index = "xyz".index(axis)
    plane = np.median(xyz[:, index]) if coordinate is None else coordinate
    keep = (xyz[:, index] - dx / 2 <= plane) & (xyz[:, index] + dx / 2 > plane)
    dims = [a for a in range(3) if a != index]
    unit = KPC / 1e5
    centers = xyz[keep][:, dims] / unit
    half = dx[keep] / unit / 2
    vertices = centers[:, None, :] + half[:, None, None] * np.array(
        [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    )
    value = np.asarray(cell[field])[keep]
    value = np.log10(np.maximum(value, np.finfo(float).tiny)) if log else value
    fig, ax = plt.subplots(figsize=(6, 5))
    collection = PolyCollection(
        vertices, array=value, cmap="magma", edgecolors="none", rasterized=True
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    fig.colorbar(collection, ax=ax, label=("log10 " if log else "") + str(field))
    ax.set_xlabel("xyz"[dims[0]] + " [kpc]")
    ax.set_ylabel("xyz"[dims[1]] + " [kpc]")
    if result is not None:
        # Match the half-open containing-leaf slice selection, including when
        # the slice lies exactly on a mesh face (e.g. z=boxlen/2).
        ok = (
            result.shock
            & (result.pos[:, index] - result.dx / 2 <= plane)
            & (result.pos[:, index] + result.dx / 2 > plane)
        )
        p = result.center_position[ok][:, dims] / unit
        ax.scatter(p[:, 0], p[:, 1], s=3, color="cyan", label="Profile centers")
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_distribution(result, reference_mach=None, bins=80):
    import matplotlib.pyplot as plt

    rows = np.flatnonzero(result.shock)
    mach = result.mach[rows]
    values = mach if reference_mach is None else mach / reference_mach
    # This is a detection-area proxy, not the archived benchmark triangle mesh.
    weights = result.dx[rows] ** 2 / np.max(abs(result.normal[rows]), axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(values):
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            density=True,
            histtype="step",
            color="#197447",
        )
    else:
        ax.text(0.5, 0.5, "No accepted profiles", ha="center", transform=ax.transAxes)
    if reference_mach is not None:
        ax.axvline(1, color=".4", ls=":")
    ax.set_xlabel("Mach" if reference_mach is None else "Mach / expected Mach")
    ax.set_ylabel("Normalized detection-area density")
    fig.tight_layout()
    return fig, ax


def expected_reference(folder, meta):
    """Evaluation ONLY: custom plane ICs or approximate strong Sedov solution.

    All lengths returned here are in km, matching ``load_snapshot``. These
    predictions are never passed to the detector. Unknown namelists return None.
    """
    from ._ramses_format import nml_value

    folder = Path(folder)
    path = folder / "namelist.txt"
    if not path.exists():
        return None
    text = path.read_text()
    g = 5 / 3
    time = float(meta["time"])
    unit = float(meta["unit_l"]) / 1e5
    if "plane_normal" in text.lower():
        normal = nml_value(text, "plane_normal", 3)
        normal /= np.linalg.norm(normal)
        mach = nml_value(text, "mach_upstream")
        rho = nml_value(text, "rho_upstream")
        p = nml_value(text, "p_upstream")
        center = (
            nml_value(text, "plane_center", 3)
            + (nml_value(text, "plane_offset") + mach * np.sqrt(g * p / rho) * time)
            * normal
        )
        return dict(
            kind="exact_RH_plane",
            mach=mach,
            center=center * unit,
            normal=normal,
            radius=None,
        )
    if "'sedov'" in text.lower() or '"sedov"' in text.lower():
        if time <= 0:
            return None
        center = np.array([nml_value(text, a + "_center", 2)[1] for a in "xyz"])
        rho = nml_value(text, "d_region", 2)[0]
        pressure = nml_value(text, "p_region", 2)
        energy = pressure[1] / (g - 1)
        radius = 1.15167 * (energy * time * time / rho) ** 0.2
        mach = 0.4 * radius / time / np.sqrt(g * pressure[0] / rho)
        return dict(
            kind="strong_Sedov_approximation",
            mach=mach,
            center=center * unit,
            normal=None,
            radius=radius * unit,
        )
    return None


def reference_errors(position, width, mach, reference):
    delta = np.asarray(position) - reference["center"]
    if reference["radius"] is None:
        distance = np.abs(delta @ reference["normal"])
    else:
        distance = np.abs(np.linalg.norm(delta, axis=1) - reference["radius"])
    return np.abs(np.asarray(mach) / reference["mach"] - 1), distance / np.asarray(
        width
    )


def weighted_quantile(values, weights, q):
    values = np.asarray(values)
    weights = np.asarray(weights)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return np.nan
    order = np.argsort(values[good])
    v = values[good][order]
    w = weights[good][order]
    return np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), v)


def summarize_profiles(path):
    """Small dictionary for a notebook table; NPZ is loaded without pickle."""
    with np.load(path, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata_json"]))
        n = len(a["mach"])
        row = dict(
            case=meta.get("case", ""),
            output=meta.get("output", ""),
            time=meta.get("time", np.nan),
            levelmin=meta.get("levelmin", np.nan),
            levelmax=meta.get("levelmax", np.nan),
            count=n,
            mach_median=np.median(a["mach"]) if n else np.nan,
            min_width=np.min(a["width"]) if n else np.nan,
            bounded_fraction=np.mean(a["center_constraint_active"]) if n else np.nan,
            quality_clear_count=int(a['quality_clear'].sum()) if 'quality_clear' in a else np.nan,
            quality_clear_fraction=float(a['quality_clear'].mean()) if n and 'quality_clear' in a else np.nan,
        )
        ref = meta.get("reference")
        row["reference_kind"] = ref["kind"] if ref else "unavailable"
        row["geometry"] = (
            "sphere"
            if ref and ref["radius"] is not None
            else "plane" if ref else "unknown"
        )
        row["angle_deg"] = (
            float(np.degrees(np.arccos(np.clip(abs(ref["normal"][0]), 0, 1))))
            if ref and ref["normal"] is not None
            else np.nan
        )
        unit = meta.get("unit_l", 1.0) / 1e5
        row["sphere_center"] = (
            ",".join(f"{v/unit:.4g}" for v in ref["center"])
            if ref and ref["radius"] is not None
            else ""
        )
        row["mach_expected"] = ref["mach"] if ref else np.nan
        row["mach_error_p95"] = row["position_error_p95_dx"] = np.nan
        if n and ref:
            em, ex = reference_errors(a["position"], a["width"], a["mach"], ref)
            w = a["width"] ** 2 / np.max(np.abs(a["normal"]), axis=1)
            row["mach_error_p95"] = weighted_quantile(em, w, 0.95)
            row["position_error_p95_dx"] = weighted_quantile(ex, w, 0.95)
        return row
