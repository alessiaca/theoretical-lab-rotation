import numpy as np
from build_model import build_goal_loop
from utils import simulation, interactive_simulation, visualize_stimulation_results
import matplotlib.pyplot as plt
# Script simulating the behaviour of the model

# Open issues: Before trauma (no amygdala activation), what are the parameters such that goal 1 is chosen over goal 2?

# Show interactive simulation with fixed weights
units = build_goal_loop()
interactive_simulation(units, 150)

print("debug")