"""The native extensions must match the Python and NumPy doing the import."""

import errno
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from shocktest import _merger_neighbors
from shocktest._native import extension_directory, load_extension
from shocktest.core import _shockfinder
from tools.build_extensions import install_extensions


def test_native_modules_load_from_active_environment():
    expected = extension_directory().resolve()
    assert Path(_shockfinder.__file__).resolve().parent == expected
    assert Path(_merger_neighbors.__file__).resolve().parent == expected
    assert importlib.import_module("shocktest._shockfinder") is _shockfinder
    assert importlib.import_module("shocktest._merger_neighbors") is _merger_neighbors


def test_numpy_upgrade_selects_another_build_directory(monkeypatch):
    original = extension_directory()
    monkeypatch.setattr(np, "__version__", "999.0")
    assert extension_directory() != original


def test_missing_environment_build_never_uses_package_root_so(tmp_path, monkeypatch):
    monkeypatch.setattr("shocktest._native.extension_directory", lambda: tmp_path)
    with pytest.raises(ImportError, match="not built for Python"):
        load_extension("_shockfinder")


def test_direct_import_cannot_fall_back_to_old_so():
    code = (
        "import numpy as np; "
        "np.__version__ = '999.0'; "
        "import shocktest; "
        "assert shocktest._shockfinder is None; "
        "assert shocktest._merger_neighbors is None; "
        "import importlib; "
        "importlib.import_module('shocktest._merger_neighbors')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "None in sys.modules" in result.stderr


def test_install_extensions_handles_separate_build_filesystem(tmp_path, monkeypatch):
    source_dir = tmp_path / "temporary-build"
    destination = tmp_path / "repository-build"
    source_dir.mkdir()
    destination.mkdir()
    built = []
    for name in ("_shockfinder.so", "_merger_neighbors.so"):
        artifact = source_dir / name
        artifact.write_bytes(f"new {name}".encode())
        target = destination / name
        target.write_bytes(b"old")
        built.append((artifact, target))

    real_replace = os.replace

    def replace_on_same_filesystem(source, target):
        if not Path(source).is_relative_to(destination):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source, target)

    monkeypatch.setattr("tools.build_extensions.os.replace", replace_on_same_filesystem)
    install_extensions(built, destination)

    for artifact, target in built:
        assert target.read_bytes() == artifact.read_bytes()
    assert not list(destination.glob(".f2py-install-*"))
