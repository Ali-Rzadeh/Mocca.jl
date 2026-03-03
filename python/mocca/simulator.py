from dataclasses import dataclass
import numpy as np
from .model import MoccaModel
from .forces import (
    AdsorptionBC,
    PressurisationBC,
    BlowdownBC,
    EvacuationBC,
    pressure_pressurisation,
    pressure_blowdown,
    pressure_evacuation,
)
from .physics import compute_equilibrium, compute_ki


@dataclass
class MoccaCase:
    model: MoccaModel
    state0: dict
    timesteps: list[float]
    forces: list[object]


def _apply_bc_pressure(force, p_cell, t):
    if isinstance(force, AdsorptionBC):
        return force.PH, force.T_feed, force.y_feed
    if isinstance(force, PressurisationBC):
        return pressure_pressurisation(force, t), force.T_feed, force.y_feed
    if isinstance(force, BlowdownBC):
        return pressure_blowdown(force, t), None, None
    if isinstance(force, EvacuationBC):
        return pressure_evacuation(force, t), None, None
    return p_cell, None, None


def simulate_process(case: MoccaCase):
    model = case.model
    p = model.system.p
    n = model.ncells
    dx = model.dx

    P = case.state0["Pressure"].copy()
    T = case.state0["Temperature"].copy()
    Tw = case.state0["WallTemperature"].copy()
    y = case.state0["y"].copy()
    q = case.state0["AdsorbedConcentration"].copy()

    states = []
    t_acc = 0.0

    for dt, force in zip(case.timesteps, case.forces):
        cTot = P / (p.R * T)
        c = y * cTot[None, :]

        qstar = np.zeros_like(q)
        k = np.zeros_like(q)
        for i in range(n):
            qstar[:, i] = compute_equilibrium(model.system, c[:, i], T[i])
            k[:, i] = compute_ki(model.system, c[:, i], qstar[:, i])

        dqdt = k * (qstar - q)
        q += dt * dqdt

        # crude advection for gas composition and pressure
        v = np.full(n + 1, p.v_feed if isinstance(force, AdsorptionBC) else 0.0)
        if isinstance(force, (PressurisationBC, BlowdownBC, EvacuationBC)):
            v[:] = 0.2

        for comp in range(2):
            flux = np.zeros(n + 1)
            flux[1:-1] = v[1:-1] * c[comp, :-1]
            if isinstance(force, (AdsorptionBC, PressurisationBC)):
                _, _, yb = _apply_bc_pressure(force, P[0], t_acc)
                flux[0] = v[0] * (yb[comp] * cTot[0])
            else:
                flux[0] = v[0] * c[comp, 0]
            flux[-1] = v[-1] * c[comp, -1]
            c[comp, :] += -(dt / dx) * (flux[1:] - flux[:-1]) - dt * dqdt[comp, :]

        cTot = np.maximum(c.sum(axis=0), 1e-12)
        y = c / cTot[None, :]
        y = np.clip(y, 1e-12, 1 - 1e-12)
        y /= y.sum(axis=0, keepdims=True)

        # pressure relaxation to boundary condition
        pbc, Tbc, _ = _apply_bc_pressure(force, P[0], t_acc)
        P[0] += 0.2 * (pbc - P[0])
        if isinstance(force, AdsorptionBC):
            P[-1] += 0.2 * (force.PH - P[-1])
        elif isinstance(force, BlowdownBC):
            P[-1] += 0.2 * (pbc - P[-1])
        P[1:] += 0.01 * (P[:-1] - P[1:])

        # simple thermal model with adsorption heat release
        dH = (p.qsbi * (p.dUbi - p.R * p.T0) + p.qsdi * (p.dUdi - p.R * p.T0)) / (p.qsbi[0] + p.qsdi[0])
        heat = (dH[:, None] * dqdt).sum(axis=0) * model.solid_volume
        T += dt * (0.001 * np.gradient(np.gradient(T, dx), dx) - 1e-7 * heat)
        if Tbc is not None:
            T[0] += 0.05 * (Tbc - T[0])
        Tw += dt * (0.001 * np.gradient(np.gradient(Tw, dx), dx) + 0.01 * (T - Tw) - 0.005 * (Tw - p.T_a))

        t_acc += dt
        states.append(
            {
                "Pressure": P.copy(),
                "Temperature": T.copy(),
                "WallTemperature": Tw.copy(),
                "y": y.copy(),
                "AdsorbedConcentration": q.copy(),
                "time": t_acc,
            }
        )

    return states, case.timesteps
