"""Build only the optional accelerated profile geometry for this interpreter."""

from pathlib import Path
import importlib.util, os, subprocess, sys, sysconfig
from build_extensions import fortran_build_environment, install_extensions

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "native_paths", ROOT / "shocktest/_native.py"
)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)
work = ROOT / "build/profile_geometry"
work.mkdir(parents=True, exist_ok=True)
flags = (
    os.environ.get("SHOCKFINDER_F90FLAGS", "-O3 -fopenmp")
    + " -ffp-contract=off -ffree-line-length-none"
)
backend = "meson" if sys.version_info >= (3, 12) else "distutils"
cmd = [
    sys.executable,
    "-m",
    "numpy.f2py",
    "-c",
    str(ROOT / "shocktest/fortran/profile_geometry.f90"),
    "-m",
    "_profile_geometry",
    "--backend",
    backend,
    "--f90flags=" + flags,
    "only:",
    "build_index",
    "locate_batch",
    "sample_batch",
    "derivatives",
    "derivatives_at",
    "fit_windows",
    "mass_root",
    ":",
]
library = os.environ.get("SHOCKFINDER_OPENMP_LIB", "gomp")
if library:
    cmd.append("-l" + library)
env = fortran_build_environment(flags)
env["TMPDIR"] = str(work)
with (work / "build.log").open("w") as log:
    subprocess.run(
        cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
    )
artifact = work / ("_profile_geometry" + sysconfig.get_config_var("EXT_SUFFIX"))
install_extensions(
    [(artifact, paths.extension_path("_profile_geometry"))], paths.extension_directory()
)
print("Built", paths.extension_path("_profile_geometry"))
