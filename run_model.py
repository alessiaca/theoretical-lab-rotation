import numpy as np
from build_model import build_model
from utils import simulation, interactive_simulation, visualize_stimulation_results, reset_units
import matplotlib.pyplot as plt

# Script simulating the behaviour of the model

# Show interactive simulation
units = build_model()
#interactive_simulation(units, 150)

# Run the simulation several times and record the chosen action and weights
plot = False
n_stim = 40
dt = 1
t_max = 40000
t = np.arange(0, t_max, dt)
binary_switches = {"Cue":[]} # Switch on cue right at the beginning
one_won = np.zeros((n_stim, 1)); two_won = np.zeros((n_stim, 1))
weights = np.zeros((n_stim, 1))
for i_stim in np.arange(n_stim):

    # Simulate the activity of the network
    simulation(units, t, dt, binary_switches=binary_switches)

    # Get the connection weights and save the last weight
    dls_cue_weights = np.array([connection.weight_history for connection in units["DLS_1"].connections if
                                connection.input_unit.name == "Cue"]).T
    #weights[i_stim] = dls_cue_weights[-1][0]

    # Save whether 1 or 2 won the competition
    one_won[i_stim] = np.any(np.array(units["MC_1"].activity_history)[-10000:, 1] > 0.5)
    two_won[i_stim] = np.any(np.array(units["MC_2"].activity_history)[-10000:, 1] > 0.5)

    # Plot the activity
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        for name in ["Cue","PL_1","PL_2","MC_1","MC_2","NAc_1","NAc_2","BLA_1","BLA_2"]:
            plt.plot(np.array(units[name].activity_history)[:,1],label=name)
            if name == "Cue":
                plt.plot(np.array(units[name].activity_history)[:, 2], label="Trace")
        plt.legend()
        # Plot the weights
        plt.subplot(1, 2, 2)
        plt.plot(dls_cue_weights)
        plt.show()

    # Reset the units
    reset_units(units)

    # Print the results of the simulation
    print(f"{i_stim + 1} of {n_stim} with result: {one_won[i_stim]} vs.{two_won[i_stim]} and weight {weights[i_stim]}")

# Print the results of the simulation (which action was chosen)
approach_percentage = np.round(np.sum(one_won)/n_stim, 2) * 100
avoid_percentage = np.round(np.sum(two_won)/n_stim, 2) * 100
print(f"Approach: {approach_percentage} Escape: {avoid_percentage}")

# Plot the weight over the trials
colors = plt.cm.Reds(np.linspace(0.2, 0.5, 3))
plt.figure()
plt.subplot(1, 1, 1)
plt.scatter(range(n_stim),weights,c=colors[-1])
plt.ylabel("$\Delta w_{DLS,Cue}$", fontSize=12)
plt.xlabel("# Trial", fontSize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Adjust layout to make room for the table:

# Add the results of the stimulation as a table
table = plt.table(cellText=[[f"{approach_percentage} %"], [f"{avoid_percentage} %"], [str(n_stim)]],
                      rowLabels=['Approach percentage', 'Escape percentage', 'Total number of simulations'],
                      loc='top', colWidths = [0.3], rowColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.subplots_adjust(left=0.2,right=0.2,top=0.2,bottom=0.2)
plt.show()
print("debug")