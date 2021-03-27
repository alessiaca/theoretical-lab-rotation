import numpy as np
from scipy.integrate import odeint

# First attempts to implement the model by Mannella et al.
# Is it correct to start like this or is there a better way?
# How to implement the excitatory/inhibitory units? How to implement the learning process?


def pos_sat(x):
    """"Returns 0 if x <= 0, else x"""
    return np.max([x,0])


class Unit:
    """Class for the units in the model"""

    # Initialize the unit
    def __init__(self, type, tau=300, theta=0, sigma=1, u_0=10, tau_i=None, u_i_0=10, iota=None, delta=None):
        self.type = type  # Define the type of the unit: "leaky","leaky onset", "excitatory", "dopaminergic" or "inhibitory"
        self.input_units = []  # Initialize an empty list of input units
        self.u = u_0 # Initialize the potential of the unit
        self.tau = tau  # Define the time constant of the unit
        self.theta = theta  # Define the steepness of the hyperbolic function for the unit activation
        self.sigma = sigma  # Define the activation threshold

        # Define the time constant and the initial potential for the inhibitory population of leaky onset units
        if type == "leaky onset":
            self.tau_i = tau_i
            self.u_i = u_i_0

        # Define the parameters weighting the input dependent and independent of dopamine for a dopaminergic unit
        if type == "dopaminergic":
            self.iota = iota
            self.delta = delta

    # Compute the potential of the unit (dependent on the type)
    def func(self):

        if self.type == "leaky":
            # Check if the unit is connected to a dopaminergic unit
            dopa_unit = [unit[0] for unit in self.input_units if unit[0].type == "dopaminergic"]
            if dopa_unit:
                dopa_unit = dopa_unit[0]
                return (-self.u + (dopa_unit.iota + dopa_unit.delta * dopa_unit.activation()) * self.input()) / self.tau
            else:
                return (-self.u + self.input()) / self.tau

        elif self.type == "leaky onset":
            u = (-self.u + pos_sat(self.input() - self.u_i)) / self.tau
            u_i = (-self.u_i + self.input()) / self.tau_i
            return u, u_i

    # Add a new input unit with a specific weight
    def add_input_unit(self, input_unit, input_weight=0):
        self.input_units.append([input_unit, input_weight])

    # Given the potential and the parameters, compute the activation of the unit
    def activation(self):
        return pos_sat(np.tanh(self.sigma * (self.u - self.theta)))

    # Compute the input to the unit given the input units and connection weights
    def input(self):
        return np.sum([input_unit.activation() * input_weight for input_unit, input_weight
                       in self.input_units if input_unit.type != "dopaminergic"])


# Initialize some units (Goal loop)
units = {}
units["CSa"] = Unit("leaky onset", tau=500, tau_i=500)
units["CSb"] = Unit("leaky onset", tau=500, tau_i=500)
units["USa"] = Unit("leaky onset", tau=500, tau_i=500)
units["USb"] = Unit("leaky onset", tau=500, tau_i=500)
units["LH"] = Unit("leaky onset", tau=100, tau_i=500)
units["VTA"] = Unit("dopaminergic", iota=0.8, delta=4)
units["NAc_1"] = Unit("leaky")
units["NAc_2"] = Unit("leaky")
units["STNv_1"] = Unit("leaky")
units["STNv_2"] = Unit("leaky")
units["SNpr_1"] = Unit("leaky")
units["SNpr_2"] = Unit("leaky")
units["DM_1"] = Unit("leaky")
units["DM_2"] = Unit("leaky")
units["PL_1"] = Unit("leaky", tau=2000, sigma=20, theta=0.8)
units["PL_2"] = Unit("leaky", tau=2000, sigma=20, theta=0.8)


# Connect some
units["VTA"].add_input_unit(units["LH"],1)
units["LH"].add_input_unit(units["CSa"],1)
units["CSa"].add_input_unit(units["CSb"],1)


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
print("test")