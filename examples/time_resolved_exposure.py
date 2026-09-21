"""Runnable analytic example: two tracked snapshots of a merger-shock candidate.

Run from the repository root:
    python -m examples.time_resolved_exposure

Replace these analytic frames with ShockFrame.from_samples(...) and persistent
IDs from your simulation tracking. Per-snapshot catalog IDs are not tracks.
"""
import numpy as np
from shocktest import (
    ShockFrame, MergerEvent, ExposureAccumulator,
    attribute_merger_shocks, integrate_galaxy_exposure,
)

KPC_KM = 3.0856775814913673e16
MYR_S = 1.e6 * 365.25 * 86400.


def run_example():
    def frame(time_myr, radius_kpc):
        return ShockFrame(
            time_s=time_myr*MYR_S,
            patch_id=np.array([10]), surface_id=np.array([1]),
            pos_km=np.array([[radius_kpc*KPC_KM, 0., 0.]]),
            normal=np.array([[1., 0., 0.]]),
            radius_km=np.array([200.*KPC_KM]),
            half_width_km=np.array([10.*KPC_KM]),
            mach=np.array([3.]), flux=np.array([1.e39]),
            tracking_source='analytic outward translating patch',
        )
    previous, current = frame(0, 100), frame(100, 400)
    merger = MergerEvent(
        event_id='analytic-merger', start_s=0., end_s=100.*MYR_S,
        center_km=(0., 0., 0.), axis=(1., 0., 0.),
        radius_min_km=50.*KPC_KM, radius_max_km=1000.*KPC_KM,
        source='synthetic event for validation; replace with merger-tree evidence',
    )
    attribution = attribute_merger_shocks(previous, current, [merger])
    position = np.array([[250.*KPC_KM, 0., 0.]])
    interval = integrate_galaxy_exposure(
        np.array([42]), position, position, previous, current,
        attributions=attribution,
    )
    history = ExposureAccumulator()
    history.add(interval)
    return attribution, interval, history


if __name__ == '__main__':
    attribution, interval, history = run_example()
    print('Attribution:', attribution[1].status)
    print('Crossing time [Myr]:', interval.records[0].crossing_times_s[0]/MYR_S)
    print('Galaxy 42:', history.summaries[42])
