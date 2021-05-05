import numpy as np
from build_model import build_model
from utils import simulation, interactive_simulation, visualize_stimulation_results, reset_units, neg_sat
import matplotlib.pyplot as plt

# Script simulating the behaviour of the model

# Decide which state of the model to simulate
trauma = True
learning = True
learning_type = "approach" # or "escape"
units = build_model(trauma=trauma, learning=learning, learning_type=learning_type)

# Run the simulation several times and record the chosen action and weights
plot = False  # Plot the activity of each trial
n_stim = 40  # Number of trials
dt = 1
t_max = 40000  # Length of a trial
t = np.arange(0, t_max, dt)
binary_switches = {"Cue":[]} # Switch on cue right at the beginning
# Initialize arrays storing the chosen actions and weights after each trial
one_won = np.zeros((n_stim, 1))
two_won = np.zeros((n_stim, 1))
weights = np.zeros((n_stim, 2))

for i_stim in np.arange(n_stim):

    # Simulate the activity of the network
    simulation(units, t, dt, binary_switches=binary_switches)

    # Get the connection weights and save the last weight
    if learning:
        dls_cue_weights = np.zeros((int(t_max*(i_stim+1)),2))
        if learning_type == "approach":
            names = ["DLS_1","DLS_2"]
        else:
            names = ["DLS_2"]
        for i,name in enumerate(names):
            dls_cue_weights[:, i] = np.array([connection.weight_history for connection in units[name].connections if
                                        connection.input_unit.name == "Cue"]).flatten()

            weights[i_stim, i] = dls_cue_weights[-1, i]

    # Save whether 1 or 2 won the competition
    one_won[i_stim] = np.any(np.array(units["MC_1"].activity_history)[-10000:, 1] > 0.7)
    two_won[i_stim] = np.any(np.array(units["MC_2"].activity_history)[-10000:, 1] > 0.7)

    # Plot the activity
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        for name in ["Cue","PL_1","PL_2","MC_1","MC_2","DLS_1","DLS_2","BLA_1","BLA_2", "CeA"]:
            plt.plot(np.array(units[name].activity_history)[:,1],label=name)
            if name == "Cue":
                plt.plot(np.array(units[name].activity_history)[:, 0], label="Cue Potential")
                plt.plot(np.array(units[name].activity_history)[:, 2], label="Trace")
        plt.legend()
        # Plot the weights
        if learning:
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
plt.scatter(range(n_stim),weights[:,0],c=colors[-1], label="Approach")
plt.scatter(range(n_stim),weights[:,1],c=colors[-2], label="Escape")
plt.ylabel("$\Delta w_{DLS,Cue}$", fontSize=12)
plt.xlabel("# Trial", fontSize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add the results of the stimulation as a table
table = plt.table(cellText=[[f"{approach_percentage} %"], [f"{avoid_percentage} %"], [str(n_stim)]],
                      rowLabels=['Approach percentage', 'Escape percentage', 'Total number of simulations'],
                      loc='top', colWidths = [0.3], rowColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.legend()
plt.show()

print("debug")