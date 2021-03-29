import numpy as np
import matplotlib.pyplot as plt

# Next steps: Integrate trace and learning
# Connect all neurons
# Visualize behaviour better 


def pos_sat(x):
    """"Returns 0 if x <= 0, else x"""
    return np.max([x,0])


class Unit:
    """Class for the units in the model"""

    # Initialize the unit
    def __init__(self, type, tau=300, thres=0, sigma=1, potential_0=0, tau_i=None, potential_i_0=0, dopa_de=None, dopa_in=None):
        self.type = type  # Define the type of the unit: "leaky","leaky onset", "binary" or "dopaminergic"
        self.input_units = []  # Initialize an empty list of input units
        self.potential = potential_0 # Initialize the potential of the unit
        self.firing_rate = 0
        self.tau = tau  # Define the time constant of the unit
        self.thres = thres  # Define the steepness of the hyperbolic function for the unit activation
        self.sigma = sigma  # Define the activation threshold
        self.activity_history = []  # Initalize an array that stores the potential and firing rate at each point in time

        # Define the time constant and the initial potential for the inhibitory population of leaky onset units
        if type == "leaky onset":
            self.tau_i = tau_i
            self.potential_i = potential_i_0

        # Define the parameters weighting the input dependent and independent of dopamine for a dopaminergic unit
        if type == "dopaminergic":
            self.dopa_in = dopa_in
            self.dopa_de = dopa_de

    # Add a new input units with a specific weight
    def add_input_units(self, input_units_and_weights):
        self.input_units.extend(input_units_and_weights)

    # Update the potential of the unit
    def integrate(self, dt):

        if self.type == "leaky" or self.type == "dopaminergic":
            potential_change = (-self.potential + self.get_input_weight() * self.input()) / self.tau
            self.potential = self.potential + potential_change * dt

        elif self.type == "leaky onset":
            potential_change = (-self.potential + pos_sat(self.input() - self.potential_i)) / self.tau
            potential_i_change = (-self.potential_i + self.input()) / self.tau_i
            self.potential = self.potential + potential_change * dt
            self.potential_i = self.potential_i + potential_i_change * dt

    # Get the weight of the input (dependent on dopaminergic input units)
    def get_input_weight(self):
        # Check if the unit is connected to a dopaminergic unit
        dopa_input_unit = [unit[0] for unit in self.input_units if unit[0].type == "dopaminergic"]
        if dopa_input_unit:
            dopa_input_unit = dopa_input_unit[0]
            input_weight = dopa_input_unit.dopa_in + dopa_input_unit.dopa_de * dopa_input_unit.firing_rate
        else:
            input_weight = 1
        return input_weight

    # Given the potential and the parameters, compute the firing rate of the neuron
    def update_firing_rate(self):
        if self.type != "binary":
            self.firing_rate = pos_sat(np.tanh(self.sigma * (self.potential - self.thres)))

    # Compute the input to the unit given the input units and connection weights
    def input(self):
        return np.sum([input_unit.firing_rate * input_weight for input_unit, input_weight
                       in self.input_units if input_unit.type != "dopaminergic"])

    # Switch the binary units of and on
    def switch_activation_binary_units(self):
        self.firing_rate = 1 if self.firing_rate == 0 else 0

    # Update the potential of the model and compute its firing rate, save the values
    def activity(self, dt=1):
        self.integrate(dt=dt)
        self.update_firing_rate()
        self.activity_history.append([self.potential, self.firing_rate])


def check_units():
    """Create a binary, leaky and leaky onset unit to check whether they behave as expected
    Turn the binary unit on and off and plot the firing rates"""

    units = {}
    units["Binary"] = Unit("binary"); units["Binary"].switch_activation_binary_units()
    units["Leaky"] = Unit("leaky"); units["Dopaminergic"] = Unit("dopaminergic")
    units["Leaky Onset"] = Unit("leaky onset", tau=500, tau_i=500)
    units["Leaky"].add_input_units([[units["Binary"], 1]]); units["Dopaminergic"].add_input_units([[units["Binary"], 1]])
    units["Leaky Onset"].add_input_units([[units["Binary"], 1]])

    t = np.arange(0, 10000, 1)
    for i_t in range(1, len(t)):
        if i_t == len(t) / 2:  # Switch food off again to check w
            units["Binary"].switch_activation_binary_units()
        for i_u, unit in enumerate(units.values()):
            unit.activity(1)

    plt.figure()
    for i_u, (name, unit) in enumerate(units.items()):
        firing_rate = np.array(unit.activity_history)[:,1]
        plt.subplot(len(units), 1, i_u+1)
        plt.plot(firing_rate)
        plt.title(name, fontsize=7)
        plt.ylabel("Firing rate", fontsize=5)
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
    plt.subplots_adjust(hspace=1)
    plt.show()

check_units()

def build_and_connect_model_units():

    # Initialize the units of the model (start with the goal loop)
    units = {}
    units["FoodA"] = Unit("binary"); units["FoodB"] = Unit("binary"); units["SatA"] = Unit("binary"); units["SatB"] = Unit("binary")
    units["CSa"] = Unit("leaky onset", tau=500, tau_i=500); units["CSb"] = Unit("leaky onset", tau=500, tau_i=500)
    units["USa"] = Unit("leaky onset", tau=500, tau_i=500); units["USb"] = Unit("leaky onset", tau=500, tau_i=500)
    units["LH"] = Unit("leaky onset", tau=100, tau_i=500); units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=4)
    units["NAc_1"] = Unit("leaky"); units["NAc_2"] = Unit("leaky")
    units["STNv_1"] = Unit("leaky"); units["STNv_2"] = Unit("leaky")
    units["SNpr_1"] = Unit("leaky"); units["SNpr_2"] = Unit("leaky")
    units["DM_1"] = Unit("leaky"); units["DM_2"] = Unit("leaky")
    units["PL_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.8); units["PL_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.8)

    # Connect the units
    #units["CSa"].add_input_units([[units["FoodA"], 1], [units["FoodB"], 1], [units["SatA"], -1], [units["SatB"], -1]])
    #units["CSb"].add_input_units([[units["FoodA"], 1], [units["FoodB"], 1], [units["SatA"], -1], [units["SatB"], -1]])
    units["NAc_1"].add_input_units([[units["FoodA"],1]])
    units["CSa"].add_input_units([[units["FoodA"],1]])
    units["NAc_2"].add_input_units([[units["SatA"],-1]])
