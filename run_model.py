import numpy as np
from build_model import build_plastic_model, build_fixed_model, build_test_model, build_BLA_model, Unit
from utils import simulation, interactive_simulation
import matplotlib.pyplot as plt
# Script simulating the behaviour of the model

# Build the model
units = build_BLA_model()

# Define the times at which the binary units should be switched
binary_switches = {}
t = np.arange(0, 1000000, 1) # Time array of simulation
time_on = 10000
lever_on_switches = np.linspace(0,len(t)-time_on, 4,dtype=int)
lever_off_switches = lever_on_switches + time_on
binary_switches["Lever"] = np.vstack((lever_on_switches,lever_off_switches))

# Run the model
simulation(units,t, binary_switches)

# Visualize the activity
plt.plot(np.array(units["CS"].activity_history)[:,2])

# Visualize the activity of the model in an interactive way
interactive_simulation(units, 100)

# Run the model


print("debug")