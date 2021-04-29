import numpy as np
from utils import pos_sat, neg_sat

# Script defining the Unit class and building the model

class Unit:
    """Class for the units in the model"""

    #######################################################################################################
    # Initialize the unit
    def __init__(self, type, name=None, tau=300, thres=0, sigma=1, potential_0=0, tau_i=None, potential_i_0=0, dopa_de=None,
                 dopa_in=None, noise_coeff=0, eta_str = 0.05, thres_da_str = 0.9, thres_str = 0.9, thres_inp_str = 0.9):
        self.type = type  # Define the type of the unit: "leaky","leaky onset", "binary" or "dopaminergic"
        self.name = name
        self.connections = []  # Initialize an empty list of input units
        self.potential = potential_0 # Initialize the potential of the unit
        self.firing_rate = 0
        self.trace = 0
        self.trace_change = 0
        self.tau = tau  # Define the time constant of the unit
        self.thres = thres  # Define the steepness of the hyperbolic function for the unit activation
        self.sigma = sigma  # Define the activation threshold
        self.activity_history = []#[[self.potential, self.firing_rate, self.trace]]  # Initialize an array that stores the potential and firing rate at each point in time
        self.tau_i = tau_i
        self.potential_i = potential_i_0
        self.noise = 0
        self.noise_coeff = noise_coeff # Noise coefficient
        self.dopa_in = dopa_in
        self.dopa_de = dopa_de
        self.eta_str = eta_str
        self.thres_da_str = thres_da_str
        self.thres_str = thres_str
        self.thres_inp_str = thres_inp_str
        # Hardcode these values for BLA learning
        self.tau_trace = 500
        self.alpha = 10**10 # Why doesn't it work if alpha is 10**10??
        self.thres_da_bla = 0.7
        self.max_w_bla = 2
        self.eta_bla = 0.08

    #######################################################################################################
    # Define a subclass for connections a neuron can have
    class Connection:
        def __init__(self,input_unit, weight, type):
            self.input_unit = input_unit
            self.weight = weight
            self.weight_history = []
            self.type = type

    #######################################################################################################
    # Add new connections: List of input units and connection weights
    def add_connections(self, connections):
        for connection in connections:
            # If no connection type is given append "fixed", give the possibility to make it "plastic"
            input_unit, weight, type = connection + ["fixed"] * (3 - len(connection))
            self.connections.append(self.Connection(input_unit, weight, type))

    #######################################################################################################
    # Update the potential, trace and weights
    def integrate(self, dt):

        # Update the potential
        if self.type == "leaky" or self.type == "dopaminergic":
            potential_change = (-self.potential + self.get_input_weight() * self.input()) / self.tau
            self.potential = self.potential + potential_change * dt
        elif self.type == "leaky onset":
            potential_change = (-self.potential + pos_sat(self.input() - self.potential_i)) / self.tau
            potential_i_change = (-self.potential_i + self.input()) / self.tau_i
            self.potential = self.potential + potential_change * dt
            self.potential_i = self.potential_i + potential_i_change * dt

        # Update the weights - only for units that receive dopaminergic input
        dopa_input_unit = self.get_dopa_input_unit()
        if dopa_input_unit:
            for connection in self.connections:
                # Update only plastic connections
                if connection.type == "plastic":
                    # Leaky onset units (BLA learning)
                    if self.type == "leaky onset":
                        weight_change = self.eta_bla * pos_sat(dopa_input_unit.firing_rate - self.thres_da_bla) * \
                                        pos_sat(self.trace_change) * neg_sat(connection.input_unit.trace_change) * \
                                        (self.max_w_bla - connection.weight)
                    # Leaky units (Striatal learning)
                    elif self.type == "leaky":
                        weight_change = self.eta_str * pos_sat(dopa_input_unit.firing_rate - self.thres_da_str) * \
                                        pos_sat(self.firing_rate - self.thres_str) *\
                                        pos_sat(connection.input_unit.firing_rate  - self.thres_inp_str)
                    connection.weight = connection.weight + weight_change * dt
                    # Save the connection weight in an array
                    connection.weight_history.append(connection.weight)

        # Update the noise
        noise_change = (-self.noise + self.noise_coeff * np.random.uniform(-0.5,0.5)) / 80
        self.noise = self.noise + noise_change * dt

    #######################################################################################################
    # Return the dopaminergic input unit if there is one, else none
    def get_dopa_input_unit(self):
        dopa_input_unit = [connection.input_unit for connection in self.connections
                           if connection.input_unit.type == "dopaminergic"]
        if dopa_input_unit:
            return dopa_input_unit[0]
        else:
            return None

    #######################################################################################################
    # Get the weight of the input (dependent on dopaminergic input units)
    def get_input_weight(self):
        # Check if the unit is connected to a dopaminergic unit
        dopa_input_unit = self.get_dopa_input_unit()
        if dopa_input_unit:
            input_weight = dopa_input_unit.dopa_in + dopa_input_unit.dopa_de * dopa_input_unit.firing_rate
        else:
            input_weight = 1
        return input_weight

    #######################################################################################################
    # Given the potential and the parameters, compute the firing rate of the neuron
    def update_firing_rate(self):
        if self.type != "binary":
            self.firing_rate = pos_sat(np.tanh(self.sigma * (self.potential - self.thres)))


    #######################################################################################################
    # Compute the input to the unit given the input units and connection weights (plus the noise)
    def input(self):
        input = 0
        for connection in self.connections:
            input_unit = connection.input_unit
            # If the input comes from the unit on the same level and the unit was already updated, take the firing rate
            # from the last time step -> Simultaneous update on one level
            if input_unit.name[:2] == self.name[:2] and len(input_unit.activity_history) > len(self.activity_history):
                input += input_unit.activity_history[-1][1] * connection.weight
            else:
                input += input_unit.firing_rate * connection.weight

        return input + self.noise


    #######################################################################################################
    # Switch the binary units of and on
    def switch_activation_binary_units(self):
        self.firing_rate = 1 if self.firing_rate == 0 else 0

    #######################################################################################################
    # Update the potential of the model and compute its firing rate, save the values
    def activity(self, dt=1):
        self.integrate(dt=dt)
        self.update_firing_rate()
        self.activity_history.append([self.potential, self.firing_rate, self.trace])


###########################################################################################################
def build_model():

    # Initialize the units of the model
    units = {}
    units["Cue"] = Unit("binary")

    # Goal loop
    units["NAc_1"] = Unit("leaky"); units["NAc_2"] = Unit("leaky")
    units["STNv_1"] = Unit("leaky"); units["STNv_2"] = Unit("leaky")
    units["SNpr_1"] = Unit("leaky"); units["SNpr_2"] = Unit("leaky")
    units["DM_1"] = Unit("leaky", noise_coeff=20); units["DM_2"] = Unit("leaky", noise_coeff=20)
    units["PL_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.1); units["PL_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.1)

    # Action loop
    units["DLS_1"] = Unit("leaky"); units["DLS_2"] = Unit("leaky")
    units["STNdl_1"] = Unit("leaky"); units["STNdl_2"] = Unit("leaky")
    units["GPi_1"] = Unit("leaky"); units["GPi_2"] = Unit("leaky")
    units["MGV_1"] = Unit("leaky", noise_coeff=0.25); units["MGV_2"] = Unit("leaky", noise_coeff=0.25)
    units["MC_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.1); units["MC_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.1)

    # Connect the units - fixed at maximum connection weight

    # Goal loop
    units["NAc_1"].add_connections([[units["Cue"], 2], [units["PL_1"], 0.75]])
    units["NAc_2"].add_connections([[units["PL_2"], 0.75]])
    units["STNv_1"].add_connections([[units["PL_1"], 1]])
    units["STNv_2"].add_connections([[units["PL_2"], 1]])
    units["SNpr_1"].add_connections([[units["NAc_1"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["SNpr_2"].add_connections([[units["NAc_2"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["DM_1"].add_connections([[units["SNpr_1"], -1.5], [units["DM_2"], -1.8], [units["DM_1"], 0.8]])
    units["DM_2"].add_connections([[units["SNpr_2"], -1.5], [units["DM_1"], -1.8], [units["DM_2"], 0.8]])
    units["PL_1"].add_connections([[units["DM_1"], 1]]); units["PL_2"].add_connections([[units["DM_2"], 1]])
    #units["PL_1"].add_connections([[units["MC_1"], 1]]); units["PL_2"].add_connections([[units["MC_2"], 1]])

    # Action loop
    units["DLS_1"].add_connections([[units["Cue"], 2], [units["MC_1"], 1]])
    units["DLS_2"].add_connections([[units["MC_2"], 1]])
    units["STNdl_1"].add_connections([[units["MC_1"], 1]])
    units["STNdl_2"].add_connections([[units["MC_2"], 1]])
    units["GPi_1"].add_connections([[units["DLS_1"], -3], [units["STNdl_1"], 2], [units["STNdl_2"], 2]])
    units["GPi_2"].add_connections([[units["DLS_2"], -3], [units["STNdl_1"], 2], [units["STNdl_2"], 2]])
    units["MGV_1"].add_connections([[units["GPi_1"], -1.5], [units["MGV_1"], 0.8], [units["MGV_2"], -0.8]])
    units["MGV_2"].add_connections([[units["GPi_2"], -1.5], [units["MGV_1"], -0.8], [units["MGV_2"], 0.8]])
    units["MC_1"].add_connections([[units["MGV_1"], 1]]); units["MC_2"].add_connections([[units["MGV_2"], 1]])
    units["MC_1"].add_connections([[units["PL_1"], 0.2]]); units["MC_2"].add_connections([[units["PL_2"], 0.2]])

    # Add the names to the class objects
    for name,unit in units.items():
        unit.name = name
    return units


