from dataclasses import dataclass
import numpy as np


@dataclass
class AdsorptionBC:
    y_feed: np.ndarray
    PH: float
    v_feed: float
    T_feed: float


@dataclass
class PressurisationBC:
    y_feed: np.ndarray
    PH: float
    PL: float
    lambd: float
    T_feed: float
    cycle_time: float
    previous_step_end: float


@dataclass
class BlowdownBC:
    PH: float
    PI: float
    lambd: float
    cycle_time: float
    previous_step_end: float


@dataclass
class EvacuationBC:
    PL: float
    PI: float
    lambd: float
    cycle_time: float
    previous_step_end: float


def pressure_pressurisation(force: PressurisationBC, time: float) -> float:
    cycle_no = np.floor(time / force.cycle_time)
    t0 = cycle_no * force.cycle_time + force.previous_step_end
    t = time - t0
    return force.PH - (force.PH - force.PL) * np.exp(-force.lambd * t)


def pressure_blowdown(force: BlowdownBC, time: float) -> float:
    cycle_no = np.floor(time / force.cycle_time)
    t0 = cycle_no * force.cycle_time + force.previous_step_end
    t = time - t0
    return force.PI + (force.PH - force.PI) * np.exp(-force.lambd * t)


def pressure_evacuation(force: EvacuationBC, time: float) -> float:
    eps = 1e-6
    cycle_no = np.floor(time / (force.cycle_time + eps))
    t0 = cycle_no * force.cycle_time + force.previous_step_end
    t = time - t0
    return force.PL + (force.PI - force.PL) * np.exp(-force.lambd * t)
