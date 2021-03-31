import numpy as np
import matplotlib.pyplot as plt

# Next steps:
# Get to work the instrumental learning 



def pos_sat(x):
    """"Returns 0 if x <= 0, else x"""
    return np.max([x,0])
def neg_sat(x):
    """"Returns 0 if x >= 0, else x"""
    return np.min([x,0])


class Unit:
    """Class for the units in the model"""

    # Initialize the unit
    def __init__(self, type, tau=300, thres=0, sigma=1, potential_0=0, tau_i=None, potential_i_0=0, dopa_de=None,
                 dopa_in=None):
        self.type = type  # Define the type of the unit: "leaky","leaky onset", "binary" or "dopaminergic"
        self.connections = []  # Initialize an empty list of input units
        self.potential = potential_0 # Initialize the potential of the unit
        self.firing_rate = 0
        self.trace = 0
        self.tau = tau  # Define the time constant of the unit
        self.thres = thres  # Define the steepness of the hyperbolic function for the unit activation
        self.sigma = sigma  # Define the activation threshold
        self.activity_history = []  # Initialize an array that stores the potential and firing rate at each point in time
        self.tau_i = tau_i
        self.potential_i = potential_i_0
        self.dopa_in = dopa_in
        self.dopa_de = dopa_de
        self.tau_trace = 500
        self.alpha = 10**10
        self.thres_da_bla = 0.7
        self.max_w_bla = 2
        self.eta_bla = 0.08
        self.trace_change = 0


    # Define a subclass for connections a neuron can have
    class Connection:
        def __init__(self,input_unit, weight):
            self.input_unit = input_unit
            self.weight = weight
            self.weight_history = []
            self.type = "fixed"
            if weight == 0:
                self.type = "plastic"

    # Add new connections: List fo input units and connection weight
    def add_connections(self, connections):
        for input_unit, weight in connections:
            self.connections.append(self.Connection(input_unit, weight))

    # Update the trace, weights and potential of the unit
    def integrate(self, dt):

        # Update the trace
        self.trace_change = ((-self.trace + self.alpha * self.firing_rate) / self.tau_trace)
        self.trace = self.trace + self.trace_change * dt

        # Update the weights (if the unit receives dopaminergic input)
        dopa_input_unit = self.get_dopa_input_unit()
        if dopa_input_unit:
            for connection in self.connections:
                # Update only the connections to non-binary units
                if connection.type == "plastic":
                    weight_change = self.eta_bla * pos_sat(dopa_input_unit.firing_rate - self.thres_da_bla) * \
                                    pos_sat(self.trace_change) * neg_sat(connection.input_unit.trace_change) * \
                                    (self.max_w_bla - connection.weight)
                    connection.weight = connection.weight + weight_change * dt
                    # Save the connection weight in an array
                    connection.weight_history.append(connection.weight)


        if self.type == "leaky" or self.type == "dopaminergic":
            potential_change = (-self.potential + self.get_input_weight() * self.input()) / self.tau
            self.potential = self.potential + potential_change * dt

        elif self.type == "leaky onset":
            potential_change = (-self.potential + pos_sat(self.input() - self.potential_i)) / self.tau
            potential_i_change = (-self.potential_i + self.input()) / self.tau_i
            self.potential = self.potential + potential_change * dt
            self.potential_i = self.potential_i + potential_i_change * dt

    # Return the dopaminergic input unit if there is one, else none
    def get_dopa_input_unit(self):
        dopa_input_unit = [connection.input_unit for connection in self.connections
                           if connection.input_unit.type == "dopaminergic"]
        if dopa_input_unit:
            return dopa_input_unit[0]
        else:
            return None

    # Get the weight of the input (dependent on dopaminergic input units)
    def get_input_weight(self):
        # Check if the unit is connected to a dopaminergic unit
        dopa_input_unit = self.get_dopa_input_unit()
        if dopa_input_unit:
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
        return np.sum([connection.input_unit.firing_rate * connection.weight for connection
                       in self.connections if connection.input_unit.type != "dopaminergic"])

    # Switch the binary units of and on
    def switch_activation_binary_units(self):
        self.firing_rate = 1 if self.firing_rate == 0 else 0

    # Update the potential of the model and compute its firing rate, save the values
    def activity(self, dt=1):
        self.integrate(dt=dt)
        self.update_firing_rate()
        self.activity_history.append([self.potential, self.firing_rate, self.trace])


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

#check_units()

def build_and_connect_model_units():

    # Initialize the units of the model (start with the goal loop)
    units = {}
    units["Lever"] = Unit("binary")
    units["FoodA"] = Unit("binary"); units["FoodB"] = Unit("binary")
    units["SatA"] = Unit("binary"); units["SatB"] = Unit("binary")
    units["CSa"] = Unit("leaky onset", tau=500, tau_i=500); units["CSb"] = Unit("leaky onset", tau=500, tau_i=500)
    units["USa"] = Unit("leaky onset", tau=500, tau_i=500); units["USb"] = Unit("leaky onset", tau=500, tau_i=500)
    units["LH"] = Unit("leaky onset", tau=100, tau_i=500)
    units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=4)
    units["NAc_1"] = Unit("leaky"); units["NAc_2"] = Unit("leaky")
    units["STNv_1"] = Unit("leaky"); units["STNv_2"] = Unit("leaky")
    units["SNpr_1"] = Unit("leaky"); units["SNpr_2"] = Unit("leaky")
    units["DM_1"] = Unit("leaky"); units["DM_2"] = Unit("leaky")
    units["PL_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.8); units["PL_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.8)

    # Connect the units
    # BLA units connected to themselves?
    units["CSa"].add_connections([[units["Lever"], 5], [units["CSb"], 0], [units["USa"], 0], [units["USb"], 0], [units["VTA"], 1]])
    units["CSb"].add_connections([[units["Lever"], 5], [units["CSa"], 0], [units["USa"], 0], [units["USb"], 0], [units["VTA"], 1]])
    units["USa"].add_connections([[units["FoodA"], 5], [units["FoodB"], 5], [units["SatA"], -10], [units["SatB"], -10],
                                 [units["CSa"], 0], [units["CSb"], 0], [units["USb"], 0], [units["VTA"], 1]])
    units["USb"].add_connections([[units["FoodA"], 5], [units["FoodB"], 5], [units["SatA"], -10], [units["SatB"], -10],
                                 [units["CSa"], 0], [units["CSb"], 0], [units["USa"], 0], [units["VTA"], 1]])
    units["LH"].add_connections([[units["FoodA"], 10], [units["FoodB"], 5], [units["USa"], 5], [units["USb"], 5]])
    units["VTA"].add_connections([[units["LH"], 20]])
    units["NAc_1"].add_connections([[units["VTA"], 1], [units["USa"], 0], [units["USb"], 0], [units["PL_1"], 1]])
    units["NAc_2"].add_connections([[units["VTA"], 1], [units["USa"], 0], [units["USb"], 0], [units["PL_2"], 1]])
    units["STNv_1"].add_connections([[units["PL_1"], 1]])
    units["STNv_2"].add_connections([[units["PL_2"], 1]])
    units["SNpr_1"].add_connections([[units["NAc_1"], -3], [units["STNv_1"], -2], [units["STNv_2"], -2]])
    units["SNpr_2"].add_connections([[units["NAc_2"], -3], [units["STNv_1"], -2], [units["STNv_2"], -2]])
    units["DM_1"].add_connections([[units["SNpr_1"], 1], [units["DM_1"], 1], [units["DM_2"], -0.8]])
    units["DM_2"].add_connections([[units["SNpr_2"], 1],  [units["DM_1"], -0.8], [units["DM_2"], 1]])
    units["PL_1"].add_connections([[units["DM_1"], 1]])
    units["PL_2"].add_connections([[units["DM_2"], 1]])

    return units

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
    units["US"].add_connections([[units["Food"], 5], [units["CS"], 0], [units["VTA"], 1]])
    units["LH"].add_connections([[units["Food"], 10], [units["US"], 5]])
    units["VTA"].add_connections([[units["LH"], 20]])

    # Run the model
    t = np.arange(0, 10000, 1)
    lever_on_switches = np.linspace(0,len(t),100,dtype=int)
    food_on_switches = lever_on_switches + 10
    for i_t in range(len(t)):
        if i_t in lever_on_switches:  # Switch food off again
            units["Lever"].switch_activation_binary_units()
        elif i_t in lever_on_switches+5:
            units["Lever"].switch_activation_binary_units()
        if i_t in food_on_switches:  # Switch food off again
            units["Food"].switch_activation_binary_units()
        elif i_t in food_on_switches + 5:
            units["Food"].switch_activation_binary_units()
        for i_u, unit in enumerate(units.values()):
            unit.activity(1)

    # Plot the behaviour
    plt.plot(np.array(units["Lever"].activity_history)[:,1])
    plt.show()

test_instrumental_training()
print("hu")