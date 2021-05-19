import numpy as np
from build_model import build_model
from utils import simulation, interactive_simulation, reset_units
import matplotlib.pyplot as plt

# Script simulating the behaviour of the model

# Initialize a plastic model
units = build_model()

# Interactive simulation
#interactive_simulation(units,200)

# Simulate the behaviour of the model in the different stages
stages = ["before trauma", "after trauma"]
stages = ["after trauma"]

# Decide whether to plot the activity after each trial (for visual inspection)
plot = False
# List of units whose firing rate should be plotted
plot_units_firing_rate = ["NAc_1","NAc_2", "PL_1", "PL_2", "DM_1", "DM_2", "BLA", "SNpr_1", "SNpr_2", "STNv_1", "STNv_2"]

# Initialize the parameters needed for the simulation
t_max = 10000  # Length of a trial
t = np.arange(0, t_max)
n_stim = 40  # Number of simulations per stage

for stage in stages:

    print("Starting stage: " + stage)

    # Prepare the stage
    if stage == "after trauma":
        binary_switches = {"hippocampus": [0]}
    else:
        binary_switches = {"hippocampus": []}

    # Initialize arrays storing the chosen actions and weights after each trial
    one_won = np.zeros((n_stim, 1))
    two_won = np.zeros((n_stim, 1))
    weights = np.zeros((n_stim, 2))  # Weights from NAc_1 and NAc_2

    for i_stim in np.arange(n_stim):

        # Simulate the activity of the network
        simulation(units, t, binary_switches=binary_switches)

        # Get the connection weights and save the last weight
        dls_cue_weights = np.zeros((t_max, 2))
        for i, name in enumerate(["NAc_1", "NAc_2"]):
            dls_cue_weights[:, i] = np.array([connection.weight_history for connection in units[name].connections if
                                              connection.input_unit.name == "BLA"]).flatten()
            weights[i_stim, i] = dls_cue_weights[-1, i]

        # Save whether 1 or 2 won the competition
        one_won[i_stim] = np.any(np.array(units["PL_1"].activity_history)[:, 1] > 0.8)
        two_won[i_stim] = np.any(np.array(units["PL_2"].activity_history)[:, 1] > 0.8)

        # Plot the activity if needed
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            for name in plot_units_firing_rate:
                plt.plot(np.array(units[name].activity_history)[:,1], label=name)
                # For the Cue plot also the potential and the trace
                if name == "BLA":
                    plt.plot(np.array(units[name].activity_history)[:, 2], label="BLA Trace")
            plt.legend()
            # Plot the weights
            plt.subplot(1, 2, 2)
            plt.plot(dls_cue_weights, label=["NAc_1_BLA", "NAc_2_BLA"])
            plt.show()

        # Reset the units
        reset_units(units)

        # Print the results of the trial
        print(f"{i_stim + 1} of {n_stim} with result: {one_won[i_stim]} vs.{two_won[i_stim]} and weight {weights[i_stim]}")

    # Print the results of the trials in a stage (which action was chosen)
    approach_percentage = np.round(np.sum(one_won)/n_stim, 2) * 100
    escape_percentage = np.round(np.sum(two_won)/n_stim, 2) * 100
    print(f"Stage: {stage} Approach: {approach_percentage} Escape: {escape_percentage}")

    # Plot the weight over the trials
    colors = plt.cm.Reds(np.linspace(0.2, 0.5, 3))
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(range(n_stim), weights[:, 0], c=colors[-1], label="Approach")
    plt.scatter(range(n_stim), weights[:, 1], c=colors[-2], label="Escape")
    plt.ylabel("$\Delta w_{DLS,Cue}$", fontSize=12)
    plt.xlabel("# Trial", fontSize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add the results of the stimulation as a table
    table = plt.table(cellText=[[f"{approach_percentage} %"], [f"{escape_percentage} %"], [str(n_stim)]],
                      rowLabels=['Approach percentage', 'Escape percentage', 'Total number of simulations'],
                      loc='top', colWidths=[0.3], rowColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.subplots_adjust(top=.83)
    plt.suptitle(stage)
    plt.legend()
plt.show()

print("debug")