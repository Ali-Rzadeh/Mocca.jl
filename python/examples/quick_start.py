from pathlib import Path
import mocca

repo = Path(__file__).resolve().parents[2]
inp = repo / "models" / "json" / "haghpanah_DCB_input_simple.json"
constants, info = mocca.parse_input(str(inp))
info.num_cycles = 1
info.stage_types = ["adsorption"]
info.stage_durations = [200.0]
info.maxdt = 1.0

case = mocca.setup_mocca_case(constants, info)
states, timesteps = mocca.simulate_process(case)

out_csv = repo / "results_python_dcb.csv"
mocca.export_cell_results(str(out_csv), case, states, timesteps)
fig = mocca.plot_outlet(case, states, timesteps)
fig.savefig(repo / "results_python_dcb.png", dpi=150)
print(f"Wrote {out_csv}")
