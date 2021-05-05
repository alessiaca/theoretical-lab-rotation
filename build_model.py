import numpy as np
from utils import pos_sat, neg_sat

# Script defining the Unit class and building the model

class Unit:
    """Class for the units in the model"""

    #######################################################################################################
    # Initialize the unit
    def __init__(self, type, name=None, tau=300, thres=0, sigma=1,noise_coeff=0):
        self.type = type  # Define the type of the unit: "leaky" or "binary"
        self.name = name
        self.connections = []  # Initialize an empty list of input units
        self.potential = 0 # Initialize the potential of the unit
        self.firing_rate = 0
        self.tau = tau  # Define the time constant of the unit
        self.thres = thres  # Define the steepness of the hyperbolic function for the unit activation
        self.sigma = sigma  # Define the activation threshold
        self.noise = 0
        self.noise_coeff = noise_coeff # Noise coefficient
        self.tau_noise = 80
        self.trace = 0
        self.tau_trace = 20000
        self.thres_trace = 0.8  # Threshold when amplification coefficient is added
        self.alpha = 20000
        self.eta = 0.0003  # Learning rate (between cue and DLS)
        self.thres_DLS = 0.7
        self.thres_Cue = 0.2
        self.thres_CeA = 1
        self.not_crossed_trace= True
        self.max_weight = 1
        self.activity_history = [[0, 0, 0], [0, 0, 0]]  # Initialize an array that stores the potential and firing rate at each point in time


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
        potential_change = (-self.potential + self.input()) / self.tau
        if self.name == "Cue":
            potential_change = neg_sat(self.input(),-0.7,1) / self.tau
        self.potential = self.potential + potential_change * dt

        # Update the trace - only for the Cue
        # Add the amplification coefficient only once when the activation threshold is passed (for the first time)
        if self.name == "Cue":
            if self.not_crossed_trace and self.firing_rate > self.thres_trace and \
                    np.all(self.firing_rate > np.array(self.activity_history)[:-1, 1]):
                trace_change = (-self.trace + self.firing_rate * self.alpha) / self.tau_trace
                self.not_crossed_trace = False
            else:
                trace_change = -self.trace / self.tau_trace
            self.trace = self.trace + trace_change * dt

        # Update the noise - only for the thalamus
        if self.name[:2] in ["MG", "DM"]:
            noise_change = (-self.noise + self.noise_coeff * np.random.uniform(-0.5,0.5)) / self.tau_noise
            self.noise = self.noise + noise_change * dt

        # Update the weight
        for connection in self.connections:
            # Update only plastic connections
            if connection.type == "plastic":
                # Get the CeA unit which controls learning
                CeA_unit = [connection.input_unit for connection in self.connections if connection.input_unit.name == "CeA"][0]
                # Compute the weight update
                weight_change = self.eta * pos_sat(self.firing_rate - self.thres_DLS) * \
                                pos_sat(connection.input_unit.activity_history[-1][2] - self.thres_Cue) * \
                                neg_sat(CeA_unit.firing_rate - self.thres_CeA) * (self.max_weight - connection.weight)
                connection.weight = connection.weight + weight_change * dt
                # Save the connection weight in an array
                connection.weight_history.append(connection.weight)

    #######################################################################################################
    # Given the potential and the parameters, compute the firing rate of the neuron
    def update_firing_rate(self):
        if self.type != "binary":
            self.firing_rate = pos_sat(np.tanh(self.sigma * (self.potential - self.thres)))
        else: # Switch binary unit off if potential (accumulated inhibitory input) is too low
            if self.potential < -0.7:
                self.firing_rate = 0


    #######################################################################################################
    # Compute the input to the unit given the input units and connection weights (plus the noise)
    def input(self):
        input = 0
        for connection in self.connections:
            input_unit = connection.input_unit
            # If the input comes from the unit on the same level and the unit was already updated, take the firing rate
            # from the last time step -> Simultaneous update on one level
            if input_unit.name[:2] == self.name[:2] and len(input_unit.activity_history) > len(self.activity_history):
                input += input_unit.activity_history[-2][1] * connection.weight
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
    """Default state of the model: Before trauma and before learning (but with plastic connections)"""

    # Initialize the units of the model
    units = {}
    units["Cue"] = Unit("binary", tau=10000)

    # Amygdala
    units["BLA_1"] = Unit("leaky"); units["BLA_2"] = Unit("leaky")
    units["CeA"] = Unit("leaky", sigma=5, thres=0.5)

    # Goal loop
    units["NAc_1"] = Unit("leaky"); units["NAc_2"] = Unit("leaky")
    units["STNv_1"] = Unit("leaky"); units["STNv_2"] = Unit("leaky")
    units["SNpr_1"] = Unit("leaky"); units["SNpr_2"] = Unit("leaky")
    units["DM_1"] = Unit("leaky", noise_coeff=14); units["DM_2"] = Unit("leaky", noise_coeff=14)
    units["PL_1"] = Unit("leaky", tau=1000, sigma=2, thres=0); units["PL_2"] = Unit("leaky", tau=1000, sigma=2, thres=0)

    # Action loop
    units["DLS_1"] = Unit("leaky"); units["DLS_2"] = Unit("leaky")
    units["STNdl_1"] = Unit("leaky"); units["STNdl_2"] = Unit("leaky")
    units["GPi_1"] = Unit("leaky"); units["GPi_2"] = Unit("leaky")
    units["MGV_1"] = Unit("leaky", noise_coeff=2); units["MGV_2"] = Unit("leaky", noise_coeff=2)
    units["MC_1"] = Unit("leaky", tau=1000, sigma=2, thres=0); units["MC_2"] = Unit("leaky", tau=1000, sigma=2, thres=0)

    # Connect the units

    # Inhibitory input from the escape action to the cue
    units["Cue"].add_connections([[units["MC_2"], -1]])

    units["CeA"].add_connections([[units["BLA_2"], 2]])

    # Goal loop
    units["BLA_1"].add_connections([[units["Cue"], 2], [units["BLA_2"], -20]])
    units["BLA_2"].add_connections([[units["Cue"], 0]])
    units["NAc_1"].add_connections([[units["BLA_1"], 2], [units["PL_1"], 1]])
    units["NAc_2"].add_connections([[units["BLA_2"], 2], [units["PL_2"], 1]])
    units["STNv_1"].add_connections([[units["PL_1"], 1]])
    units["STNv_2"].add_connections([[units["PL_2"], 1]])
    units["SNpr_1"].add_connections([[units["NAc_1"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["SNpr_2"].add_connections([[units["NAc_2"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["DM_1"].add_connections([[units["SNpr_1"], -1.5], [units["DM_2"], -0.8], [units["DM_1"], 0.8]])
    units["DM_2"].add_connections([[units["SNpr_2"], -1.5], [units["DM_1"], -0.8], [units["DM_2"], 0.8]])
    units["PL_1"].add_connections([[units["DM_1"], 1]]); units["PL_2"].add_connections([[units["DM_2"], 1]])
    units["PL_1"].add_connections([[units["MC_1"], 0.2]]); units["PL_2"].add_connections([[units["MC_2"], 0.2]])

    # Action loop
    units["DLS_1"].add_connections([[units["Cue"], 0, "plastic"], [units["MC_1"], 1], [units["CeA"],  0]])
    units["DLS_2"].add_connections([[units["Cue"], 0, "plastic"], [units["MC_2"], 1], [units["CeA"],  0]])
    units["STNdl_1"].add_connections([[units["MC_1"], 1]])
    units["STNdl_2"].add_connections([[units["MC_2"], 1]])
    units["GPi_1"].add_connections([[units["DLS_1"], -3], [units["STNdl_1"], 2], [units["STNdl_2"], 2]])
    units["GPi_2"].add_connections([[units["DLS_2"], -3], [units["STNdl_1"], 2], [units["STNdl_2"], 2]])
    units["MGV_1"].add_connections([[units["GPi_1"], -1.5], [units["MGV_1"], 0.8], [units["MGV_2"], -0.8]])
    units["MGV_2"].add_connections([[units["GPi_2"], -1.5], [units["MGV_1"], -0.8], [units["MGV_2"], 0.8]])
    units["MC_1"].add_connections([[units["MGV_1"], 1]]); units["MC_2"].add_connections([[units["MGV_2"], 1]])
    units["MC_1"].add_connections([[units["PL_1"], 1]]); units["MC_2"].add_connections([[units["PL_2"], 1]])

    # Add the names to the class objects
    for name,unit in units.items():
        unit.name = name
    return units


