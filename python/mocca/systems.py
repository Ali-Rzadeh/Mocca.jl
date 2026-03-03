from dataclasses import dataclass
import numpy as np
from .constants import AdsorptionConstants


@dataclass
class TwoComponentAdsorptionSystem:
    p: AdsorptionConstants
    component_names: tuple[str, str] = ("CO2", "N2")

    @property
    def permeability(self) -> float:
        p = self.p
        return 4.0 / 150.0 * ((p.Phi / (1 - p.Phi)) ** 2) * (p.d_p / 2.0) ** 2 * p.Phi

    @property
    def dispersion(self) -> float:
        return 0.7 * self.p.D_m + 0.5 * self.p.V0_inter * self.p.d_p


def area_wall(system: TwoComponentAdsorptionSystem) -> float:
    p = system.p
    return np.pi * (p.r_out**2 - p.r_in**2)
