"""Observable limitations of ideal-gas profile measurements, not truth labels."""
from enum import IntFlag
import numpy as np


class ProfileQualityFlag(IntFlag):
    OK = 0
    ENTROPY_UNCERTAIN = 1
    CENTER_SENSITIVE = 2
    BACKGROUND_SENSITIVE = 4
    DENSITY_CLOSURE = 8
    MOMENTUM_CLOSURE = 16
    ENERGY_CLOSURE = 32
    NORMAL_BOUNDED = 64
    INCOMPLETE_TUBE = 128
    SINGLE_BACKGROUND = 256


def quality_flags(fit, full_tube=True):
    flag = ProfileQualityFlag.OK
    if fit['entropy_margin'] <= 1: flag |= ProfileQualityFlag.ENTROPY_UNCERTAIN
    if fit['center_sensitivity'] > .5: flag |= ProfileQualityFlag.CENTER_SENSITIVE
    if (np.ptp(fit['mach_range'])/fit['mach'] > .1 or np.ptp(fit['offset_range']) > .5):
        flag |= ProfileQualityFlag.BACKGROUND_SENSITIVE
    if fit['density_residual'] > .15: flag |= ProfileQualityFlag.DENSITY_CLOSURE
    if fit['momentum_residual'] > .15: flag |= ProfileQualityFlag.MOMENTUM_CLOSURE
    if fit['energy_residual'] > .15: flag |= ProfileQualityFlag.ENERGY_CLOSURE
    if fit['center_constraint_active']: flag |= ProfileQualityFlag.NORMAL_BOUNDED
    if not full_tube: flag |= ProfileQualityFlag.INCOMPLETE_TUBE
    if fit['model_count'] < 2: flag |= ProfileQualityFlag.SINGLE_BACKGROUND
    return int(flag)


def galaxy_pressure(sampling, galaxy_velocity, density_factor, side=0):
    """Return thermal/ram pressure in erg/cm3 for gas/galaxy velocities in km/s.

    density_factor converts the stored density unit to g/cm3. The caller must
    use the same physical frame for gas and galaxy. This is instantaneous local
    forcing, not evidence of stripping, merger origin or a past crossing.
    """
    rho = np.asarray(sampling['density'])[:, side] * density_factor
    relative = np.asarray(sampling['velocity'])[:, side] - np.asarray(galaxy_velocity)
    return dict(ram_pressure=rho*np.sum(relative**2,axis=-1)*1e10,
                thermal_pressure=np.asarray(sampling['pressure'])[:,side].copy())
