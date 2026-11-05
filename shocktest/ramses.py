"""Single-rank, pure-hydro RAMSES leaf output reader (AMR or uniform)."""

import numpy as np
import re
from scipy.io import FortranFile
from pathlib import Path
from ._ramses_format import info


def read_amr(folder):
    folder = Path(folder)
    meta = info(folder)
    # This reader implements the idealized non-cosmological experiment layout,
    # not a conversion of RAMSES supercomoving outputs into a physical frame.
    nml = folder / "namelist.txt"
    text = (
        "\n".join(line.split("!")[0] for line in nml.read_text().splitlines())
        if nml.exists()
        else ""
    )
    if re.search(r"\bcosmo\s*=\s*\.true\.", text, re.I) or not np.isclose(
        float(meta.get("aexp", 1.0)), 1.0, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Cosmological output requires an explicitly validated physical-coordinate/velocity adapter; this reader supports non-cosmological tests only"
        )
    level = int(meta["levelmax"])
    if int(meta["ncpu"]) != 1:
        raise ValueError("single rank required")
    index = int(folder.name.split("_")[-1])
    amr = folder / f"amr_{index:05d}.out00001"
    hydro = folder / f"hydro_{index:05d}.out00001"
    side = 2**level
    total = side**3
    # Flat order is ix * side**2 + iy * side + iz.
    field_parts = []
    position_parts = []
    level_parts = []
    with FortranFile(amr, "r") as mesh, FortranFile(hydro, "r") as fluid:
        integer = lambda: mesh.read_ints(np.int32)
        skip = lambda: mesh.read_record(np.uint8)
        ncpu = int(integer()[0])
        ndim = int(integer()[0])
        coarse = integer()
        nlevel = int(integer()[0])
        integer()  # ngridmax
        nboundary = int(integer()[0])
        integer()  # ngrid_current
        box = float(mesh.read_reals(np.float64)[0])
        for _ in range(11):  # output/time/cosmology records, through mass_sph
            skip()
        integer()
        integer()  # headl, taill
        counts = integer().reshape((ncpu, nlevel), order="F")
        integer()  # numbtot
        if nboundary:
            integer()
            integer()
            boundary_counts = integer().reshape((nboundary, nlevel), order="F")
        else:
            boundary_counts = np.empty((0, nlevel), dtype=int)
        skip()  # free memory
        ordering = mesh.read_record(np.uint8).tobytes().decode().strip()
        if ordering == "bisection":
            raise ValueError("Bisection output not supported")
        skip()  # bound_key
        for _ in range(3):
            skip()  # coarse son, flag, CPU map
        headers = [int(fluid.read_ints(np.int32)[0]) for _ in range(5)]
        gamma = float(fluid.read_reals(np.float64)[0])
        if headers != [1, 5, 3, nlevel, nboundary] or ndim != 3 or ncpu != 1:
            raise ValueError(f"Unexpected AMR/hydro headers: {headers}")
        if not np.isclose(gamma, 5 / 3, rtol=0, atol=1e-12):
            raise ValueError("Expected gamma=5/3")
        for lev in range(1, nlevel + 1):
            h = box / 2**lev
            for domain in range(ncpu + nboundary):
                count = int(
                    counts[domain, lev - 1]
                    if domain < ncpu
                    else boundary_counts[domain - ncpu, lev - 1]
                )
                centers = None
                sons = None
                if count:
                    integer()
                    integer()
                    integer()  # grid, next, prev
                    centers = np.column_stack(
                        [mesh.read_reals(np.float64) for _ in range(3)]
                    )
                    integer()  # father
                    for _ in range(6):
                        integer()
                    sons = [integer() for _ in range(8)]
                    for _ in range(16):
                        integer()  # CPU map and refinement flags
                hydro_level = int(fluid.read_ints(np.int32)[0])
                hydro_count = int(fluid.read_ints(np.int32)[0])
                if (hydro_level, hydro_count) != (lev, count):
                    raise ValueError("AMR/hydro record ordering mismatch")
                for child in range(8) if count else ():
                    values = np.column_stack(
                        [fluid.read_reals(np.float64) for _ in range(5)]
                    )
                    if domain == 0:
                        bits = np.array([(child >> axis) & 1 for axis in range(3)])
                        positions = (centers - coarse // 2) * box + (bits - 0.5) * h
                        keep = (sons[child] == 0) & np.all(
                            (positions >= 0) & (positions < box), axis=1
                        )
                        field_parts.append(values[keep])
                        position_parts.append(positions[keep])
                        level_parts.append(np.full(keep.sum(), lev, np.int32))
    fields = np.concatenate(field_parts)
    positions = np.concatenate(position_parts)
    levels = np.concatenate(level_parts)
    if not np.isfinite(fields).all() or (fields[:, [0, 4]] <= 0).any():
        raise ValueError("invalid hydro")
    if not np.isclose(np.sum(2.0 ** (-3 * levels.astype(float))), 1):
        raise ValueError("invalid leaf volume")
    return fields, positions, levels, meta


def load_snapshot(folder, *, mean_molecular_weight=0.59):
    """Return a unit-aware cell mapping and metadata, using physical K and km/s.

    Supported outputs: ncpu=1, ndim=3, five adiabatic hydro variables, gamma=5/3.
    RT/MHD/extra-energy and MPI output layouts are explicitly rejected.
    """
    from .profile_reconstruction import K_B, M_P

    fields, pos, level, meta = read_amr(folder)
    length = float(meta["unit_l"])
    time = float(meta["unit_t"])
    density = float(meta["unit_d"])
    speed = length / time
    temperature = (
        fields[:, 4] / fields[:, 0] * speed**2 * mean_molecular_weight * M_P / K_B
    )
    cell = {
        ("x", "km"): pos[:, 0] * length / 1e5,
        ("y", "km"): pos[:, 1] * length / 1e5,
        ("z", "km"): pos[:, 2] * length / 1e5,
        ("dx", "km"): float(meta["boxlen"])
        * 2.0 ** (-level.astype(float))
        * length
        / 1e5,
        ("vx", "km/s"): fields[:, 1] * speed / 1e5,
        ("vy", "km/s"): fields[:, 2] * speed / 1e5,
        ("vz", "km/s"): fields[:, 3] * speed / 1e5,
        ("T", "K"): temperature,
        ("rho", "g/cc"): fields[:, 0] * density,
        ("thermal_pressure", "erg/cm3"): fields[:, 4] * density * speed**2,
        "level": level,
    }
    meta = dict(
        meta,
        output=str(Path(folder).resolve()),
        mean_molecular_weight=mean_molecular_weight,
    )
    return cell, meta


def snapshot_finder(meta):
    from .core import ShockFinder

    finder = ShockFinder()
    finder.density_unit = "g/cc"
    finder.mean_molecular_weight = meta["mean_molecular_weight"]
    finder.thermal_pressure_field = ("thermal_pressure", "erg/cm3")
    return finder
