import numpy as np
from build_model import build_model
from utils import simulation, interactive_simulation, visualize_stimulation_results, reset_units
import matplotlib.pyplot as plt
# Script simulating the behaviour of the model

# Open issues: Before trauma (no amygdala activation), what are the parameters such that goal 1 is chosen over goal 2?

# Show interactive simulation with fixed weights
units = build_model()
#interactive_simulation(units, 150)

# Run the simulation several times and record the chosen action
n_stim = 20
dt = 1
t_max = 10000
t = np.arange(0, t_max, dt)
binary_switches = {"Cue":[0]} # Switch on cue right at the beginning
one_won = np.zeros((n_stim,1)); two_won = np.zeros((n_stim,1))
for stim in np.arange(n_stim):

    # Simulate the activity of the network
    simulation(units, t, dt, binary_switches=binary_switches)

    # Save whether 1 or 2 won the competition
    one_won[stim] = np.any(np.array(units["MC_1"].activity_history)[-10000:, 1] > 0.5)
    two_won[stim] = np.any(np.array(units["MC_2"].activity_history)[-10000:, 1] > 0.5)
    plt.figure()
    plt.subplot(2,2,1)
    # Plot the activity
    for name in ["Cue","PL_1","PL_2","MC_1","MC_2"]:
        plt.plot(np.array(units[name].activity_history)[:,1],label=name)
        if name == "Cue":
            plt.plot(np.array(units[name].activity_history)[:, 2], label="Trace")
    # Plot the weights
    dls_cue_weights = [connection.weight_history for connection in units["DLS_1"].connections if connection.input_unit.name == "Cue"]
    plt.subplot(2, 2, 2)
    plt.plot(np.array(dls_cue_weights).flatten())
    plt.legend()
    plt.show()
    # Reset the unit to 0
    reset_units(units)
    print(f"{stim + 1} of {n_stim} with result: {one_won[stim]} vs.{two_won[stim]}")

print(f"One: {np.sum(one_won)/n_stim}"); print(f"Two: {np.sum(two_won)/n_stim}")
print("debug")