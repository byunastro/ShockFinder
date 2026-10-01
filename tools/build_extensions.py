"""Build and verify the ShockFinder F2PY modules for this interpreter."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "shocktest"
SOURCES = ("shockfinder", "merger_neighbors")


def fortran_build_environment(flags: str) -> dict[str, str]:
    """Pass Fortran flags through Meson as well as F2PY's CLI option."""
    env = os.environ.copy()
    # NumPy 1.26's Meson template ignores --f90flags. Meson reads FFLAGS
    # during its fresh setup, so keep any existing flags and append ours.
    env["FFLAGS"] = " ".join(part for part in (env.get("FFLAGS", ""), flags.strip()) if part)
    return env


def install_extensions(built: list[tuple[Path, Path]], destination: Path) -> None:
    """Stage on the destination filesystem before replacing existing modules."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".f2py-install-", dir=destination) as staging:
        staged = []
        for artifact, target in built:
            staged_artifact = Path(staging) / artifact.name
            shutil.copy2(artifact, staged_artifact)
            staged.append((staged_artifact, target))
        for staged_artifact, target in staged:
            os.replace(staged_artifact, target)


def main() -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            f"NumPy is unavailable in {sys.executable}. Activate the intended "
            "environment or set PYTHON to its Python executable."
        ) from exc

    spec = importlib.util.spec_from_file_location("shocktest_native_paths", PACKAGE / "_native.py")
    assert spec is not None and spec.loader is not None
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    destination = paths.extension_directory()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"NumPy: {np.__version__} ({np.__file__})", flush=True)
    print(f"Extensions: {destination}", flush=True)

    # Build every module before replacing any extension already in use.
    with tempfile.TemporaryDirectory(prefix="shockfinder-f2py-") as temporary:
        work = Path(temporary)
        built = []
        for source in SOURCES:
            name = f"_{source}"
            flags = os.environ.get("SHOCKFINDER_F90FLAGS", "-O3 -fopenmp")
            if source == "merger_neighbors":
                # Preserve rounding at AMR contact and normal thresholds.
                flags += " " + os.environ.get(
                    "SHOCKFINDER_FP_CONTRACT_OFF_FLAG", "-ffp-contract=off"
                )
            openmp_library = os.environ.get("SHOCKFINDER_OPENMP_LIB", "gomp")
            command = [
                sys.executable, "-m", "numpy.f2py", "-c",
                str(PACKAGE / "fortran" / f"{source}.f90"),
                "-m", name, f"--f90flags={flags.strip()}",
            ]
            if openmp_library:
                command.append(f"-l{openmp_library}")
            build_env = fortran_build_environment(flags)
            print(f"Building {name} with {os.environ.get('FC', 'default Fortran compiler')}", flush=True)
            print(f"Fortran flags: {build_env['FFLAGS']}", flush=True)
            subprocess.run(command, cwd=work, check=True, env=build_env)
            artifact = work / f"{name}{suffix}"
            if not artifact.is_file():
                raise RuntimeError(f"F2PY did not produce the expected extension: {artifact}")
            built.append((artifact, destination / artifact.name))

        install_extensions(built, destination)

    # Verify both the package and native modules under the same interpreter.
    verify = (
        "import pathlib, sys, shocktest; "
        "from shocktest.core import _shockfinder; "
        "from shocktest import _merger_neighbors; "
        "root = pathlib.Path(sys.argv[1]).resolve(); "
        "built = pathlib.Path(sys.argv[2]).resolve(); "
        "assert pathlib.Path(shocktest.__file__).resolve().parent == root / 'shocktest'; "
        "assert pathlib.Path(_shockfinder.__file__).resolve().parent == built; "
        "assert pathlib.Path(_merger_neighbors.__file__).resolve().parent == built; "
        "import shocktest._shockfinder as direct_shockfinder; "
        "import shocktest._merger_neighbors as direct_merger_neighbors; "
        "assert direct_shockfinder is _shockfinder; "
        "assert direct_merger_neighbors is _merger_neighbors; "
        "print('Import verified:', shocktest.__file__, _shockfinder.__file__, _merger_neighbors.__file__)"
    )
    subprocess.run([sys.executable, "-c", verify, str(ROOT), str(destination)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
