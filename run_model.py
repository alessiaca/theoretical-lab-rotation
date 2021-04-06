import numpy as np
from build_model import build_model, build_test_model, build_BLA_model, Unit
from utils import Simulation, visualize_simulation_interactive
# Script simulating the behaviour of the model

# Build the model
units = build_model()

# Visualize the activity of the model in an interactive way
simulation = Simulation(units, 100)
visualize_simulation_interactive(simulation)