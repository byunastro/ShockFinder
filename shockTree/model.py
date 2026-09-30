"""Compact in-memory pair data and the explicitly approved reference encoding."""

from dataclasses import dataclass

import numpy as np


KPC_KM = 3.0856775814913673e16
GYR_SECONDS = 365.25 * 86400.0 * 1e9
KMS_TO_KPC_GYR = GYR_SECONDS / KPC_KM


def node_key(timestep, shock_id):
    ids = np.asarray(shock_id, dtype=np.int64)
    times = np.asarray(timestep, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= 2**32) or np.any(times < 0) or np.any(times >= 2**31):
        raise ValueError("approved node keys require 0 <= shock_id < 2**32 and 0 <= timestep < 2**31")
    return (times << np.int64(32)) | ids


def decode_key(keys):
    keys = np.asarray(keys, dtype=np.int64)
    if np.any(keys < 0):
        raise ValueError("missing references must be masked before decoding")
    return keys >> np.int64(32), keys & np.int64(2**32 - 1)


def minimum_image(vector, box):
    if box is None:
        return vector
    return vector - box * np.floor(vector / box + 0.5)


@dataclass(frozen=True)
class Metadata:
    timestep: int
    aexp: float
    time_gyr: float
    box_comoving_kpc: np.ndarray | None = None


@dataclass
class Snapshot:
    timestep: int
    aexp: float
    time_gyr: float
    ids: np.ndarray
    pos: np.ndarray  # Matching coordinates: comoving kpc, common simulation origin.
    normal: np.ndarray
    mach: np.ndarray
    cell_size: np.ndarray  # Comoving kpc.
    dissipation: np.ndarray
    velocity: np.ndarray | None = None  # Established branch displacement / physical Gyr.
    velocity_score: np.ndarray | None = None

    def __len__(self):
        return len(self.ids)

    @property
    def keys(self):
        return node_key(self.timestep, self.ids)
