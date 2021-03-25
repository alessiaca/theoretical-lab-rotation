import numpy as np
from scipy.integrate import odeint

# First attempts to implement the model by Mannella et al.
# Next steps: Implement leaky onset units


def pos_sat(x):
    """"Returns 0 if x <= 0, else x"""
    return np.max([x,0])


class Unit:

    # Initialize the unit
    def __init__(self, type, tau=300, theta=0, delta=1, u_0=10,tau_i=None,u_i_0=None):
        self.type = type  # Define the type of the unit: "leaky","leaky onset", "excitatory" or "inhibitory"
        self.input_units = []  # Initialize an empty list of input units
        self.u = u_0
        self.tau = tau  # Define time constant of the unit
        self.theta = theta  # Define the steepness of the hyperbolic function for the unit activation
        self.delta = delta  # Define the activation threshold

        if type == "leaky onset":
            self.tau_i = tau_i
            self.u_i = u_i_0

    # Compute the potential of the unit (dependent on the type)
    def func(self):
        if self.type == "leaky":
            return (-self.u + self.input()) / self.tau
        elif self.type == "leaky onset":
            u = (-self.u + pos_sat(self.input() - self.u_i)) / self.tau
            u_i = (-self.u_i + self.input()) / self.tau_i
            return u, u_i

    # Add a new input unit with a specific weight
    def add_input_unit(self, input_unit, input_weight):
        self.input_units.append([input_unit, input_weight])

    # Given the potential and the parameters, compute the activation of the unit
    def activation(self):
        return pos_sat(np.tanh(self.delta * (self.u - self.theta)))

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
            if unit.type == "leaky":
                unit.u = unit.u + unit.func() * dt
            elif unit.type == "leaky onset":
                unit.u = unit.u + unit.func()[0] * dt
                unit.u_i = unit.u_i + unit.func()[1] * dt

    return activities

test = run_model(100,1)
print("hh")