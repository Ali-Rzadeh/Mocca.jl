from dataclasses import dataclass
import numpy as np
from .systems import TwoComponentAdsorptionSystem


@dataclass
class MoccaModel:
    system: TwoComponentAdsorptionSystem
    ncells: int

    @property
    def dx(self) -> float:
        return self.system.p.L / self.ncells

    @property
    def fluid_volume(self) -> np.ndarray:
        area = np.pi * self.system.p.r_in**2
        return np.full(self.ncells, area * self.dx * self.system.p.Phi)

    @property
    def solid_volume(self) -> np.ndarray:
        area = np.pi * self.system.p.r_in**2
        return np.full(self.ncells, area * self.dx * (1 - self.system.p.Phi))
