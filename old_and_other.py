import numpy as np
import matplotlib.pyplot as plt
from build_model import Unit

def check_units():
    """Create a binary, leaky and leaky onset unit to check whether they behave as expected
    Turn the binary unit on and off and plot the firing rates"""

    units = {}
    units["Binary"] = Unit("binary"); units["Binary"].switch_activation_binary_units()
    units["Leaky"] = Unit("leaky"); units["Dopaminergic"] = Unit("dopaminergic")
    units["Leaky Onset"] = Unit("leaky onset", tau=500, tau_i=500)
    units["Leaky"].add_connections([[units["Binary"], 1]]); units["Dopaminergic"].add_connections([[units["Binary"], 1]])
    units["Leaky Onset"].add_connections([[units["Binary"], 1]])

    t = np.arange(0, 10000, 1)
    for i_t in range(1, len(t)):
        if i_t == len(t) / 2:  # Switch food off again
            units["Binary"].switch_activation_binary_units()
        for i_u, unit in enumerate(units.values()):
            unit.activity(1)

    plt.figure()
    subplots = np.arange(len(units)*3).reshape(len(units),3) + 1
    titles = ["Potential","Firing rate", "Trace"]
    for i_u, (name, unit) in enumerate(units.items()):
        activity = np.array(unit.activity_history)
        for i,title in enumerate(titles):
            plt.subplot(len(units), 3, subplots[i_u,i]); plt.plot(activity[:,i])
            plt.title(name, fontsize=7)
            plt.ylabel(title, fontsize=5)
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
    plt.subplots_adjust(hspace=1)
    plt.show()

check_units()

# Test with only one stimulus if the lever gets associated with the food (weights update as expected)
def test_instrumental_training():

    # Define the units needed
    units = {}
    units["Lever"] = Unit("binary")
    units["Food"] = Unit("binary")
    units["CS"] = Unit("leaky onset", tau=500, tau_i=500)
    units["US"] = Unit("leaky onset", tau=500, tau_i=500)
    units["LH"] = Unit("leaky onset", tau=100, tau_i=500)
    units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=4)
    # Connect them
    units["CS"].add_connections([[units["Lever"], 5], [units["US"], 0], [units["VTA"], 1]])
    units["US"].add_connections([[units["Food"], 5], [units["CS"], 0], [units["VTA"], 1]]) # Why deosn't it work that way!
    units["LH"].add_connections([[units["Food"], 10], [units["US"], 5]])
    units["VTA"].add_connections([[units["LH"], 20]])

    # Run the model
    t = np.arange(0, 1000000, 1)
    lever_on_switches = np.linspace(0,len(t), 10,dtype=int)
    food_on_switches = lever_on_switches + 5000
    for i_t in range(len(t)):
        if i_t in lever_on_switches:  # Switch food off again
            units["Lever"].switch_activation_binary_units()
        elif i_t in lever_on_switches+10000:
            units["Lever"].switch_activation_binary_units()
        if i_t in food_on_switches:  # Switch food off again
            units["Food"].switch_activation_binary_units()
        elif i_t in food_on_switches + 10000:
            units["Food"].switch_activation_binary_units()
        for i_u, unit in enumerate(units.values()):
            unit.activity(1)

    # Plot the behaviour
    plt.figure()
    subplots = np.arange(len(units) * 3).reshape(len(units), 3) + 1
    titles = ["Potential", "Firing rate", "Trace"]
    for i_u, (name, unit) in enumerate(units.items()):
        activity = np.array(unit.activity_history)
        for i, title in enumerate(titles):
            plt.subplot(len(units), 3, subplots[i_u, i])
            plt.plot(activity[:, i])
            plt.title(name, fontsize=7)
            plt.ylabel(title, fontsize=5)
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
    plt.subplots_adjust(hspace=1)

    # Plot th weights from CS(Lever) to US(Food) (should increase)
    plt.figure()
    weights = units["US"].connections[1].weight_history
    plt.plot(weights)
    plt.title("Weights from CS to US")
    plt.show()
    print("debug")

# Run the functions of interest
test_instrumental_training()
check_units()
