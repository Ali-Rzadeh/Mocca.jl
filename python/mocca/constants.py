from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class AdsorptionConstants:
    molecularMassOfCO2: float = 44.01e-3
    molecularMassOfN2: float = 28e-3
    R: float = 8.3144598
    Phi: float = 0.37
    b0: np.ndarray = field(default_factory=lambda: np.array([8.65e-7, 2.5e-6], dtype=float))
    d0: np.ndarray = field(default_factory=lambda: np.array([2.63e-8, 0.0], dtype=float))
    dUbi: np.ndarray = field(default_factory=lambda: np.array([-36641.21, -1.58e4], dtype=float))
    dUdi: np.ndarray = field(default_factory=lambda: np.array([-35690.66, 0.0], dtype=float))
    qsbi: np.ndarray = field(default_factory=lambda: np.array([3489.44, 6613.551], dtype=float))
    qsdi: np.ndarray = field(default_factory=lambda: np.array([2872.35, 0.0], dtype=float))
    eps_p: float = 0.35
    D_m: float = 1.6e-5
    tau: float = 3.0
    d_p: float = 2e-3
    V0_inter: float = 1.0
    fluid_viscosity: float = 1.72e-5
    K_z: float = 0.0903
    K_w: float = 16.0
    rho_s: float = 1130.0
    rho_g: float = 1.22638310956
    C_pg: np.ndarray = field(default_factory=lambda: np.array([697.5687, 1096.4], dtype=float))
    C_pa: np.ndarray = field(default_factory=lambda: np.array([697.5687, 1096.4], dtype=float))
    C_ps: float = 1070.0
    r_in: float = 0.1445
    r_out: float = 0.162
    h_in: float = 8.6
    h_out: float = 2.5
    rho_w: float = 7800.0
    C_pw: float = 502.0
    T0: float = 298.15
    T_a: float = 298.15
    v_feed: float = 0.37
    y_feed: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.85], dtype=float))
    p_high: float = 1e5
    p_intermediate: float = 0.2e5
    p_low: float = 0.1e5
    lambd: float = 0.5
    T_feed: float = 298.15
    L: float = 1.0
    P_init: float = 101325.0
    Tw_init: float = 298.15
    y_init: np.ndarray = field(default_factory=lambda: np.array([1e-10, 1 - 1e-10], dtype=float))


@dataclass
class ProcessInfo:
    stage_types: List[str]
    stage_durations: List[float]
    num_cycles: int
    system_type: str = "TwoComponentAdsorptionSystem"
    ncells: int = 200
    maxdt: float = 1.0
    timestep_selectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    linear_solver: str = "default"
    info_level: int = 0


def haghpanah_constants() -> tuple[AdsorptionConstants, ProcessInfo]:
    c = AdsorptionConstants()
    i = ProcessInfo(
        stage_types=["pressurisation", "adsorption", "blowdown", "evacuation"],
        stage_durations=[15.0, 15.0, 30.0, 40.0],
        num_cycles=3,
        ncells=200,
        maxdt=1.0,
    )
    return c, i
