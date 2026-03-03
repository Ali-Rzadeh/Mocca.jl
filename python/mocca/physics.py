import numpy as np
from .systems import TwoComponentAdsorptionSystem


def compute_equilibrium(system: TwoComponentAdsorptionSystem, c: np.ndarray, T: float) -> np.ndarray:
    p = system.p
    b = p.b0 * np.exp(-p.dUbi / (p.R * T))
    d = p.d0 * np.exp(-p.dUdi / (p.R * T))
    bC = float(np.dot(b, c))
    dC = float(np.dot(d, c))
    q = p.qsbi * b * c / (1 + bC) + p.qsdi * d * c / (1 + dC)
    return q


def compute_ki(system: TwoComponentAdsorptionSystem, c: np.ndarray, qstar: np.ndarray) -> np.ndarray:
    p = system.p
    D_p = p.D_m / p.tau
    r_p = p.d_p / 2.0
    return c / np.maximum(qstar, 1e-16) * 15.0 * p.eps_p * D_p / (r_p**2)
