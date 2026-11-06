"""Configuration for the inspection stage and future tree builder.

The server defaults are preserved. Pass --input-dir when inspecting local
copies. Snapshot suffixes are discovered, not generated from these patterns.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

INPUT_DIR = "/storage1/byunkh/NC_shock"
OUTPUT_PATH = "/storage1/byunkh/NC_shock/shock_tree.npz"

SNAPSHOT_START = 605
SNAPSHOT_END = 785

RESULT_PATTERN = "shock_output/result_*.pkl"
DISSIPATION_PATTERN = "shock_output/dissipation_*.pkl"
CATALOG_PATTERN = "shock_catalog/catalog_*.pkl"

# These must be established from real inputs or supplied provenance, not guessed.
SHOCK_ID_FIELD = "center_index"  # User-authorized alias; values must be preserved.
ID_REFERENCE_CONVENTION = "(timestep << 32) | center_index"  # Explicitly approved.
SNAPSHOT_METADATA_PATH = "snapshot_metadata.csv"
COORDINATE_FRAME = None
PERIODIC_BOX = None
DISSIPATION_JOIN_ID = None
CATALOG_JOIN_ID = None

# Bound reads to one chunk. Inspection never unpickles dense array payloads.
INSPECTION_CHUNK_ROWS = 1_000_000
NORMAL_TOLERANCE = 1e-8

shock_tree_dtype = np.dtype([
    ('timestep', '<i4'),
    ('aexp', '<f8'),
    ('mach', '<f8'),
    ('x', '<f8'),
    ('y', '<f8'),
    ('z', '<f8'),
    ('n', '<f8', (3,)),
    ('shock_id', '<i8'),
    ('fat', '<i8'),
    ('son', '<i8'),
    ('score_fat', '<f8'),
    ('score_son', '<f8'),
    ('first', '<i8'),
    ('last', '<i8'),
])


@dataclass
class MatchOptions:
    # Exploratory defaults, exposed here and in JSON; not calibrated probabilities.
    max_speed_kms: float = 3000.0
    cell_slack: float = 2.0
    prediction_uncertainty_fraction: float = 0.5
    minimum_displacement_kpc: float = 0.0
    max_interval_gyr: float = 0.3
    normal_min_cosine: float = 0.5
    max_mach_ratio: float = 4.0
    max_dissipation_ratio: float = 100.0
    mach_log_scale: float = 0.5
    dissipation_log_scale: float = 1.0
    w_pos: float = 0.60
    w_n: float = 0.20
    w_mach: float = 0.15
    w_dissipation: float = 0.05
    max_cost: float = 0.9
    score_temperature: float = 0.6
    margin_scale: float = 0.2
    margin_weight: float = 0.3
    nonmutual_score_factor: float = 0.8
    min_score: float = 0.15
    history_min_score: float = 0.35
    velocity_smoothing: float = 0.5
    query_chunk: int = 2048
    max_query_candidates: int = 200_000
    max_candidate_edges: int = 5_000_000

    def validate(self):
        for name in ("max_speed_kms", "max_interval_gyr", "mach_log_scale", "dissipation_log_scale",
                     "score_temperature", "margin_scale", "max_cost"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"matching.{name} must be finite and positive")
        for name in ("cell_slack", "minimum_displacement_kpc", "w_pos", "w_n", "w_mach", "w_dissipation"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"matching.{name} must be finite and nonnegative")
        if self.w_pos <= 0 or self.w_n <= 0 or self.w_mach <= 0:
            raise ValueError("position, normal, and Mach features require positive weights")
        for name in ("prediction_uncertainty_fraction", "margin_weight", "nonmutual_score_factor",
                     "min_score", "history_min_score", "velocity_smoothing"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"matching.{name} must be in [0,1]")
        if self.prediction_uncertainty_fraction == 0 or self.min_score == 1:
            raise ValueError("prediction uncertainty must be positive; min_score must be below 1")
        if not -1 <= self.normal_min_cosine < 1:
            raise ValueError("normal_min_cosine must be in [-1,1)")
        for name in ("max_mach_ratio", "max_dissipation_ratio"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 1:
                raise ValueError(f"matching.{name} must exceed 1")
        for name in ("query_chunk", "max_query_candidates", "max_candidate_edges"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"matching.{name} must be a positive integer")


@dataclass
class PipelineConfig:
    input_dir: str = INPUT_DIR
    manifest_path: str | None = None
    output_path: str = OUTPUT_PATH
    snapshot_start: int = SNAPSHOT_START
    snapshot_end: int = SNAPSHOT_END
    metadata_path: str = SNAPSHOT_METADATA_PATH
    # Explicitly required: neither the producer nor the pickles record these.
    coordinate_frame: str | None = COORDINATE_FRAME
    coordinate_origin: str | None = None
    periodic: bool | None = None
    box_comoving_kpc: list[float] | None = None
    producer_contract: str = "analyze_dense_same_call"
    catalog_mode: str = "disabled_by_producer"
    dissipation_feature: str = "flux"  # Resolution-robust power per area.
    reject_mach_inconsistent: bool = True
    chunk_rows: int = INSPECTION_CHUNK_ROWS
    diagnostics: bool = True
    diagnostic_label: str = "NewCluster"
    secondary_limit: int = 50_000
    branch_diagnostic_limit: int = 50_000
    low_confidence_score: float = 0.35
    max_plot_pairs: int = 3
    plot_samples: int = 10_000
    weights_calibrated: bool = False
    matching: MatchOptions = field(default_factory=MatchOptions)

    def validate(self, require_physics=True):
        if self.snapshot_start < 0 or self.snapshot_end < self.snapshot_start:
            raise ValueError("invalid selected snapshot range")
        if self.producer_contract != "analyze_dense_same_call":
            raise ValueError("this adapter requires the user-verified dense same-call producer; supply an explicit adapter for other inputs")
        if self.catalog_mode not in {"disabled_by_producer", "identified"}:
            raise ValueError("catalog_mode must be disabled_by_producer or identified")
        if self.dissipation_feature not in {"flux", "total"}:
            raise ValueError("dissipation_feature must be flux or total")
        if self.chunk_rows <= 0 or self.secondary_limit < 0 or self.branch_diagnostic_limit < 0:
            raise ValueError("invalid chunk or diagnostic limits")
        if self.max_plot_pairs < 0 or self.plot_samples <= 0 or not 0 <= self.low_confidence_score <= 1:
            raise ValueError("invalid diagnostic plotting/score configuration")
        self.matching.validate()
        if require_physics:
            if self.coordinate_frame not in {"physical", "comoving"}:
                raise ValueError("set coordinate_frame to physical or comoving from the input producer")
            if self.coordinate_origin not in {"same_simulation_origin", "common_unwrapped_origin"}:
                raise ValueError("declare coordinate_origin; per-snapshot recentering requires an explicit coordinate adapter")
            if not isinstance(self.periodic, bool):
                raise ValueError("set periodic explicitly; it is absent from the saved inputs")
            if self.periodic and self.coordinate_origin == "common_unwrapped_origin":
                raise ValueError("common_unwrapped_origin requires periodic=false")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path | None):
    if path is None:
        return PipelineConfig()
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    names = {item.name for item in fields(PipelineConfig)}
    unknown = values.keys() - names
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    match_values = values.pop("matching", {})
    unknown = match_values.keys() - {item.name for item in fields(MatchOptions)}
    if unknown:
        raise ValueError(f"unknown matching keys: {sorted(unknown)}")
    result = PipelineConfig(**values, matching=MatchOptions(**match_values))
    for name in ("input_dir", "output_path", "metadata_path"):
        value = Path(getattr(result, name))
        if not value.is_absolute():
            setattr(result, name, str((path.parent / value).resolve()))
    if result.manifest_path is not None and not Path(result.manifest_path).is_absolute():
        result.manifest_path = str((path.parent / result.manifest_path).resolve())
    return result
