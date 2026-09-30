"""Locate F2PY extensions built for the running Python and NumPy environment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import sysconfig
from pathlib import Path

import numpy as np


_EXTENSIONS = frozenset({"_shockfinder", "_merger_neighbors"})


def extension_directory() -> Path:
    """Keep incompatible conda environments from sharing an extension file."""
    identity = {
        "prefix": str(Path(sys.prefix).resolve()),
        "implementation": sys.implementation.name,
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "numpy_path": str(Path(np.__file__).resolve()),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return Path(__file__).resolve().parent / "_f2py_builds" / digest


def extension_path(name: str) -> Path:
    if name not in _EXTENSIONS:
        raise ValueError(f"Unknown ShockFinder extension: {name}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python did not report an extension-module suffix")
    return extension_directory() / f"{name}{suffix}"


def load_extension(name: str):
    """Load only the extension built for this interpreter and NumPy install."""
    path = extension_path(name)
    if not path.is_file():
        raise ImportError(
            f"{name} is not built for Python {sys.version.split()[0]} "
            f"and NumPy {np.__version__} in {sys.prefix}. "
            f"Run PYTHON={sys.executable} ./f2py.sh from the repository root."
        )
    fullname = f"shocktest.{name}"
    existing = sys.modules.get(fullname)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
            raise ImportError(f"{fullname} is already loaded from another build; restart Python")
        return existing
    spec = importlib.util.spec_from_file_location(fullname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {fullname} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[fullname]
        raise
    return module
