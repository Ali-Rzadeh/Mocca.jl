import csv
import json
import numpy as np
from .constants import AdsorptionConstants, ProcessInfo
from .systems import TwoComponentAdsorptionSystem
from .model import MoccaModel
from .simulator import MoccaCase
from .forces import AdsorptionBC, PressurisationBC, BlowdownBC, EvacuationBC
from .physics import compute_equilibrium


def _get_val(d):
    if isinstance(d, dict) and "value" in d:
        return d["value"]
    return d


def parse_input(inp):
    if isinstance(inp, str):
        with open(inp, "r", encoding="utf-8") as f:
            inp = json.load(f)

    c = AdsorptionConstants(
        molecularMassOfCO2=_get_val(inp["physicalConstants"]["molecularMassOfCO2"]),
        molecularMassOfN2=_get_val(inp["physicalConstants"]["molecularMassOfN2"]),
        R=_get_val(inp["physicalConstants"]["R"]),
        Phi=_get_val(inp["columnProps"]["Φ"]),
        b0=np.array(_get_val(inp["dslPars"]["b0"]), dtype=float),
        d0=np.array(_get_val(inp["dslPars"]["d0"]), dtype=float),
        dUbi=np.array(_get_val(inp["dslPars"]["ΔUbi"]), dtype=float),
        dUdi=np.array(_get_val(inp["dslPars"]["ΔUdi"]), dtype=float),
        qsbi=np.array(_get_val(inp["dslPars"]["qsbi"]), dtype=float),
        qsdi=np.array(_get_val(inp["dslPars"]["qsdi"]), dtype=float),
        eps_p=_get_val(inp["adsorbentProps"]["ϵ_p"]),
        D_m=_get_val(inp["adsorbentProps"]["D_m"]),
        tau=_get_val(inp["adsorbentProps"]["τ"]),
        d_p=_get_val(inp["adsorbentProps"]["d_p"]),
        V0_inter=_get_val(inp["adsorbentProps"]["V0_inter"]),
        rho_s=_get_val(inp["adsorbentProps"]["ρ_s"]),
        C_pa=np.array(_get_val(inp["adsorbentProps"]["C_pa"]), dtype=float),
        C_ps=_get_val(inp["adsorbentProps"]["C_ps"]),
        fluid_viscosity=_get_val(inp["feedProps"]["fluid_viscosity"]),
        rho_g=_get_val(inp["feedProps"]["ρ_g"]),
        C_pg=np.array(_get_val(inp["feedProps"]["C_pg"]), dtype=float),
        T_feed=_get_val(inp["feedProps"]["T_feed"]),
        v_feed=_get_val(inp["feedProps"]["v_feed"]),
        y_feed=np.array(_get_val(inp["feedProps"]["y_feed"]), dtype=float),
        K_z=_get_val(inp["columnProps"]["K_z"]),
        K_w=_get_val(inp["columnProps"]["K_w"]),
        r_in=_get_val(inp["columnProps"]["r_in"]),
        r_out=_get_val(inp["columnProps"]["r_out"]),
        h_in=_get_val(inp["columnProps"]["h_in"]),
        h_out=_get_val(inp["columnProps"]["h_out"]),
        rho_w=_get_val(inp["columnProps"]["ρ_w"]),
        C_pw=_get_val(inp["columnProps"]["C_pw"]),
        L=_get_val(inp["columnProps"]["L"]),
        T_a=_get_val(inp["boundaryConditions"]["T_a"]),
        p_high=_get_val(inp["boundaryConditions"]["p_high"]),
        p_intermediate=_get_val(inp["boundaryConditions"]["p_intermediate"]),
        p_low=_get_val(inp["boundaryConditions"]["p_low"]),
        lambd=_get_val(inp["boundaryConditions"]["λ"]),
        P_init=_get_val(inp["initialConditions"]["P_init"]),
        T0=_get_val(inp["initialConditions"]["T0"]),
        Tw_init=_get_val(inp["initialConditions"]["Tw_init"]),
        y_init=np.array(_get_val(inp["initialConditions"]["y_init"]), dtype=float),
    )
    i = ProcessInfo(
        stage_types=list(_get_val(inp["processSpecification"]["stage_types"])),
        stage_durations=list(_get_val(inp["processSpecification"]["stage_durations"])),
        num_cycles=int(_get_val(inp["processSpecification"]["num_cycles"])),
        system_type=_get_val(inp["simulation"]["system_type"]),
        ncells=int(_get_val(inp["simulation"]["ncells"])),
        maxdt=float(_get_val(inp["simulation"]["maxdt"])),
        timestep_selectors=_get_val(inp["simulation"]["timestep_selectors"]),
        linear_solver=_get_val(inp["solver"]["linear_solver"]),
        info_level=int(_get_val(inp["solver"]["info_level"])),
    )
    return c, i


def _setup_stage_bcs(constants: AdsorptionConstants, stage_times, stage_names, ncells):
    cycle_time = float(sum(stage_times))
    step_end = np.cumsum(stage_times)
    out = []
    for idx, name in enumerate(stage_names):
        prev = 0.0 if idx == 0 else float(step_end[idx - 1])
        if name == "pressurisation":
            out.append(PressurisationBC(constants.y_feed, constants.p_high, constants.p_low, constants.lambd, constants.T_feed, cycle_time, prev))
        elif name == "adsorption":
            out.append(AdsorptionBC(constants.y_feed, constants.p_high, constants.v_feed, constants.T_feed))
        elif name == "blowdown":
            out.append(BlowdownBC(constants.p_high, constants.p_intermediate, constants.lambd, cycle_time, prev))
        elif name == "evacuation":
            out.append(EvacuationBC(constants.p_low, constants.p_intermediate, constants.lambd, cycle_time, prev))
        else:
            raise ValueError(f"Unknown stage {name}")
    return out


def setup_mocca_case(constants: AdsorptionConstants, info: ProcessInfo):
    system = TwoComponentAdsorptionSystem(constants)
    model = MoccaModel(system=system, ncells=info.ncells)

    P = np.full(info.ncells, constants.P_init, dtype=float)
    T = np.full(info.ncells, constants.T0, dtype=float)
    Tw = np.full(info.ncells, constants.Tw_init, dtype=float)
    y = np.tile(constants.y_init[:, None], (1, info.ncells))

    ctot = P / (constants.R * T)
    q = np.zeros((2, info.ncells), dtype=float)
    for i in range(info.ncells):
        q[:, i] = compute_equilibrium(system, y[:, i] * ctot[i], T[i])

    state0 = {
        "Pressure": P,
        "Temperature": T,
        "WallTemperature": Tw,
        "y": y,
        "AdsorbedConcentration": q,
    }

    base_bcs = _setup_stage_bcs(constants, info.stage_durations, info.stage_types, info.ncells)
    timesteps = []
    forces = []
    for _ in range(info.num_cycles):
        for dur, bc in zip(info.stage_durations, base_bcs):
            nsteps = int(np.floor(dur / info.maxdt))
            timesteps.extend([info.maxdt] * nsteps)
            forces.extend([bc] * nsteps)

    return MoccaCase(model=model, state0=state0, timesteps=timesteps, forces=forces)


def export_cell_results(outputfile: str, case: MoccaCase, states, timesteps):
    with open(outputfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "P", "T", "Tw", "yCO2", "yN2", "qCO2", "qN2"])
        t = 0.0
        for st, dt in zip(states, timesteps):
            t += dt
            row = [
                t,
                st["Pressure"][-1],
                st["Temperature"][-1],
                st["WallTemperature"][-1],
                st["y"][0, -1],
                st["y"][1, -1],
                st["AdsorbedConcentration"][0, -1],
                st["AdsorbedConcentration"][1, -1],
            ]
            w.writerow(row)
