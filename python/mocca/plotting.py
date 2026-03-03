import numpy as np
import matplotlib.pyplot as plt


def plot_cell(states, timesteps, cell):
    t = np.cumsum(timesteps)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    axs = axs.ravel()
    axs[0].plot(t, [s["Pressure"][cell] for s in states]); axs[0].set_title("Pressure")
    axs[1].plot(t, [s["Temperature"][cell] for s in states]); axs[1].set_title("Temperature")
    axs[2].plot(t, [s["WallTemperature"][cell] for s in states]); axs[2].set_title("WallTemperature")
    axs[3].plot(t, [s["y"][0, cell] for s in states], label="CO2")
    axs[3].plot(t, [s["y"][1, cell] for s in states], label="N2"); axs[3].legend(); axs[3].set_title("y")
    axs[4].plot(t, [s["AdsorbedConcentration"][0, cell] for s in states], label="CO2")
    axs[4].plot(t, [s["AdsorbedConcentration"][1, cell] for s in states], label="N2"); axs[4].legend(); axs[4].set_title("q")
    fig.tight_layout()
    return fig


def plot_state(state):
    n = state["Pressure"].size
    x = np.linspace(0, 1, n)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    axs = axs.ravel()
    axs[0].plot(x, state["Pressure"]); axs[0].set_title("Pressure")
    axs[1].plot(x, state["Temperature"]); axs[1].set_title("Temperature")
    axs[2].plot(x, state["WallTemperature"]); axs[2].set_title("WallTemperature")
    axs[3].plot(x, state["y"][0, :], label="CO2")
    axs[3].plot(x, state["y"][1, :], label="N2"); axs[3].legend(); axs[3].set_title("y")
    axs[4].plot(x, state["AdsorbedConcentration"][0, :], label="CO2")
    axs[4].plot(x, state["AdsorbedConcentration"][1, :], label="N2"); axs[4].legend(); axs[4].set_title("q")
    fig.tight_layout()
    return fig


def plot_outlet(case, states, timesteps):
    return plot_cell(states, timesteps, case.model.ncells - 1)
