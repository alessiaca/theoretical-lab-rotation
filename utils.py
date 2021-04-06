# Script with help functions for the model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

##########################################################################################
# Small help functions

def pos_sat(x):
    """"Returns 0 if x <= 0, else x"""
    return np.max([x,0])


def neg_sat(x):
    """"Returns 0 if x >= 0, else -x"""
    return np.max([x*-1,0])

############################################################################################
# Interactive visualization of the models activity (based on Andreas script "Interactive.py")

class Simulation:
    """Class for the simulation of a model"""

    def __init__(self, units, step_size):
        self.units = units
        self.step_size = step_size  # Amount of new samples displayed at each time step
        self.window_size = 10000  # Number of samples displayed at the same time
        self.window = np.zeros([len(units), self.window_size])  # Array storing the activity of all units in the window
        self.step = np.zeros([len(units), self.step_size])  # Array storing the activity of all uints in the last step

    # Returns the activity of the units in the last window
    def __call__(self):
        for i in range(self.step_size):
            for j, unit in enumerate(self.units.values()):
                unit.activity()
                self.step[j, i] = unit.firing_rate
        self.window = np.hstack((self.window, self.step))
        self.window = self.window[:, -self.window_size:]
        return self.window

# Visualize the activity of the model in an interactove way
def visualize_simulation_interactive(simulation):

    units = simulation.units

    # Open the figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Add the names of the units
    names = list(simulation.units.keys())[::-1]
    plt.yticks(np.arange(len(names)),names)

    # Add a button to switch on the binary unit
    input_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
    Input = Button(input_ax, 'Binary unit', color='#fbe8a6', hovercolor='#f4976c')
    def Input_event(event):
        units["Binary"].switch_activation_binary_units()
        if Input.color == "#fbe8a6":
            Input.color = "#d2fdff"
            Input.hovercolor = "#b4def5"
        else:
            Input.color = "#fbe8a6"
            Input.hovercolor = "#f4976c"
    Input.on_clicked(Input_event)

    # Show the animated imshow
    updater = ax.imshow(simulation(), aspect='auto', vmin=0, vmax=1)
    def updatefig(*args):
        v = simulation()
        updater.set_array(v[::-1,:]) # Change the order of the units
        return updater,
    ani = animation.FuncAnimation(fig, updatefig, interval=200, blit=True)
    plt.show()
