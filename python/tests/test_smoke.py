from pathlib import Path
import mocca


def test_dcb_smoke():
    repo = Path(__file__).resolve().parents[2]
    inp = repo / "models" / "json" / "haghpanah_DCB_input_simple.json"
    constants, info = mocca.parse_input(str(inp))
    info.num_cycles = 1
    info.stage_types = ["adsorption"]
    info.stage_durations = [10.0]
    info.maxdt = 1.0
    case = mocca.setup_mocca_case(constants, info)
    states, timesteps = mocca.simulate_process(case)
    assert len(states) == len(timesteps)
    assert states[-1]["y"].shape[0] == 2
