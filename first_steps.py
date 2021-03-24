import numpy as np
from scipy.integrate import odeint

# First attempts to implement the model by Mannella et al.
# Next steps: Implement leaky onset units


class Unit:

    # Initialize the unit
    def __init__(self, type, tau=300, theta=0, delta=1, u_0=10):
        self.type = type  # Define the type of the unit: "leaky","leaky_onset", "excitatory" or "inhibitory"
        self.tau = tau  # Define time constant of the unit
        self.theta = theta  # Define the steepness of the hyperbolic function for the unit activation
        self.delta = delta  # Define the activation threshold
        self.u = u_0  # Initialize the potential of the unit
        self.input_units = []  # Initialize an empty list of input units

    # Compute the potential of the unit (dependent on the type)
    def func(self):
        if self.type == "leaky":
            return (-self.u + self.input()) / self.tau
        if self.type == "leaky_onset":


    # Add a new input unit with a specific weight
    def add_input_unit(self, input_unit, input_weight):
        self.input_units.append([input_unit, input_weight])

    # Given the potential and the parameters, compute the activation of the unit
    def activation(self):
        return np.max([np.tanh(self.delta * (self.u - self.theta)), 0])

    # Compute the input to the unit given the input units and connection weights
    def input(self):
        return np.sum([input_unit.activation() * input_weight for input_unit, input_weight in self.input_units])


# Initialize and connect units
units = {}
units["VTA"] = Unit("leaky", theta=1)
units["NAc_1"] = Unit("leaky")
units["NAc_2"] = Unit("leaky")
units["VTA"].add_input_unit(units["NAc_1"],1)
units["VTA"].add_input_unit(units["NAc_2"],1)
units["NAc_1"].add_input_unit(units["NAc_2"],1)
units["NAc_2"].add_input_unit(units["VTA"],1)


# Function to run the model
def run_model(t_max, dt):
    t = np.arange(0, t_max, dt)
    activities = np.zeros((len(units), len(t)))
    for i_t in range(1, len(t)):
        for i_u,unit in enumerate(units.values()):
            activities[i_u,i_t] = unit.u
            unit.u = unit.u + unit.func() * dt

    return activities

test = run_model(100,1)
print("hh")