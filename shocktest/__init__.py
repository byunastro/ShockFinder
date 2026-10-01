import sys as _sys

from ._native import load_extension as _load_extension


_NATIVE_IMPORT_ERRORS = {}
for _name in ("_shockfinder", "_merger_neighbors"):
    try:
        globals()[_name] = _load_extension(_name)
    except ImportError as _exc:
        _NATIVE_IMPORT_ERRORS[_name] = _exc
        globals()[_name] = None
        # Prevent a direct submodule import from finding an old .so in this directory.
        _sys.modules[f"{__name__}.{_name}"] = None


from .exposure import (
    ShockFrame, MergerEvent, MergerAttribution, ExposureRecord, ExposureInterval,
    ExposureAccumulator, attribute_merger_shocks, integrate_galaxy_exposure,
)
from .compact import ShockSamples, compact_shocks
from .core import ShockFinder, ShockResult
from .fronts import (
    shock_front_catalog, front_dtype, QUALITY_OK, QUALITY_NO_AREA,
    QUALITY_NO_DISS_RATE, QUALITY_UNDEFINED_NORMAL, QUALITY_APPROX_CONNECTIVITY,
    QUALITY_GAP_BRIDGED, QUALITY_PARTIAL_SUMMARY,
)
from .mach_validation import (
    MachValidationFlag,
    mach_from_density_jump,
    mach_from_density_ratio,
    mach_from_pressure_jump,
    mach_from_pressure_ratio,
    mach_from_temperature_jump,
    mach_from_temperature_ratio,
)
from .analysis import ShockAnalysis
from .catalog_io import (
    CATALOG_SCHEMA_VERSION,
    load_shock_catalog,
    save_shock_catalog,
    save_shock_catalog_csv,
)
from .catalog_qa import plot_catalog_quality, summarize_catalog_quality

__all__ = [
    "shock_front_catalog", "front_dtype", "QUALITY_OK", "QUALITY_NO_AREA",
    "QUALITY_NO_DISS_RATE", "QUALITY_UNDEFINED_NORMAL", "QUALITY_APPROX_CONNECTIVITY",
    "QUALITY_GAP_BRIDGED", "QUALITY_PARTIAL_SUMMARY",
    "ShockFrame", "MergerEvent", "MergerAttribution", "ExposureRecord", "ExposureInterval",
    "ExposureAccumulator", "attribute_merger_shocks", "integrate_galaxy_exposure",
    "ShockSamples",
    "compact_shocks",
    "ShockAnalysis",
    "CATALOG_SCHEMA_VERSION",
    "ShockFinder",
    "ShockResult",
    "MachValidationFlag",
    "load_shock_catalog",
    "plot_catalog_quality",
    "save_shock_catalog",
    "save_shock_catalog_csv",
    "summarize_catalog_quality",
    "mach_from_density_jump",
    "mach_from_density_ratio",
    "mach_from_pressure_jump",
    "mach_from_pressure_ratio",
    "mach_from_temperature_jump",
    "mach_from_temperature_ratio",
]
