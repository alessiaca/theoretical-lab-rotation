import numpy as np
from build_model import build_plastic_model, build_fixed_model, build_test_model, build_BLA_model, Unit, build_fixed_action_loop
from utils import simulation, interactive_simulation, visualize_stimulation_results
import matplotlib.pyplot as plt
# Script simulating the behaviour of the model

# Open issues:
# Trace: The trace decays faster than the leaky onset (Trace time coefficient to low?)
# How is the instrumental training performed? Is 0.05 dt and 15 sec the tril length? How to determine "action corresponding
# to manipulandum chosen"?
# What are the correct weights in the system? Inhibitory connection in the plot but positive weight in the table!?

# Show interactive simulation with fixed weights
units = build_fixed_action_loop()
interactive_simulation(units, 150)


# Build the model to test learning
units = build_BLA_model()

# Define the times at which the binary units should be switched
dt = 1 # sec
trial_t_max = 10000
t_max_instrumental = 60*20
binary_switches = {}
t = np.arange(0, trial_t_max, dt) # Time array of simulation
time_on_food = 1000
time_on_lever = 500
binary_switches["Lever"] = [0+dt,time_on_lever+dt]
binary_switches["Food"] = [time_on_lever+dt,time_on_lever+time_on_food+dt]

# Run the model
simulation(units,t, dt,binary_switches)

# Visualize the activity
visualize_stimulation_results(units,["US","CS","VTA"])


print("debug")