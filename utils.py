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
# Instrumental training (association between US and CS)
# def instrumental_training(units, dt=0.05, trial_length=15, n_trials=80):
#     for trial in range(n_trials):


############################################################################################
# Reset activity of units to 0
def reset_units(units):
    for unit in units.values():
        unit.potential = 0
        unit.potential_i = 0
        unit.firing_rate = 0
        unit.trace = 0
        unit.trace_change = 0


############################################################################################
# Simulation of the model
def simulation(units, t, dt, binary_switches=None):
    """binary_switches: Dictionary with names of units and times of switches (starting with 0)"""
    for i_t in t:
        for i_u, (name, unit) in enumerate(units.items()):
            unit.activity(dt)
            # Switch a binary unit on/off
            if name in binary_switches.keys() and i_t in binary_switches[name]:
                unit.switch_activation_binary_units()

############################################################################################
# Visualization of the activity of the model after simulation
def visualize_stimulation_results(units,visualize_units):
    """visualize_units: List of units whose activity should be visualized"""
    for unit_name in visualize_units:
        unit = units[unit_name]  # Get the unit to visualize
        plt.figure()
        plt.suptitle(unit_name)
        activities_names = ["Potential", "Firing rate", "Trace"]
        activities = np.array(unit.activity_history).T
        for j in range(len(activities)):
            plt.subplot(2, 2, j+1)
            plt.plot(activities[j,:])
            plt.title(activities_names[j])
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Show also the plastic weights in separate plots
        plastic_connections = [connection for connection in unit.connections if connection.type == "plastic"]
        if plastic_connections:
            plt.figure()
            plt.suptitle(unit_name)
            connection_names = [con.input_unit.name for con in plastic_connections]
            for j,connection in enumerate(plastic_connections):
                plt.subplot(2,2,j+1)
                plt.plot(connection.weight_history)
                plt.title(connection_names[j])
            plt.subplots_adjust(wspace=0.5, hspace=0.5)


    plt.show()

############################################################################################
# Interactive visualization of the models activity (based on Andreas script "Interactive.py")

class Simulation_window:
    """Class for the simulation of a model"""

    def __init__(self, units, step_size,type):
        self.units = units
        self.type = type
        self.step_size = step_size  # Amount of new samples displayed at each time step
        self.window_size = 10000  # Number of samples displayed at the same time
        self.window = np.zeros([len(units), self.window_size])  # Array storing the activity of all units in the window
        self.step = np.zeros([len(units), self.step_size])  # Array storing the activity of all uints in the last step

    # Returns the activity of the units in the last window
    def __call__(self):
        for i in range(self.step_size):
            for j, unit in enumerate(self.units.values()):
                unit.activity()
                if self.type == "firing rate":
                    self.step[j, i] = unit.firing_rate
                elif self.type == "potential":
                    self.step[j, i] = unit.potential
        self.window = np.hstack((self.window, self.step))
        self.window = self.window[:, -self.window_size:]
        return self.window

# Start an interactive simulation
def interactive_simulation(units,step_size,type="firing rate"):

    # Create a simulation object given the units and the step size of the simulation
    simulation_window = Simulation_window(units,step_size,type)

    # Open the figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Add the names of the units
    names = list(units.keys())[::-1]
    plt.yticks(np.arange(len(names)),names)

    # Get all the binary units
    binary_units = [unit for unit in units.items() if unit[1].type == "binary"]
    left_pos = list(np.linspace(0.05,0.8,len(binary_units)))

    # Add a button to switch on each binary unit
    buttons = []
    for i,(name, unit) in enumerate(binary_units):
        button_ax = plt.axes([left_pos[i], 0.05, 0.15, 0.075])
        button = Button(button_ax, name, color='#fbe8a6', hovercolor='#f4976c')
        buttons.append(button)

    # Define a function that manages what happens when a button is clicked
    def button_click_func(event):
        # Get the button that was clicked (use the position of the button for identification)
        l_pos = event.inaxes.get_position().get_points()[0, 0]
        event_i = left_pos.index(l_pos)
        binary_units[event_i][1].switch_activation_binary_units()
        button = buttons[event_i]
        if button.color == "#fbe8a6":
            button.color = "#d2fdff"
            button.hovercolor = "#b4def5"
        else:
            button.color = "#fbe8a6"
            button.hovercolor = "#f4976c"
    for button in buttons:
        button.on_clicked(button_click_func)

    # Show the animated imshow
    updater = ax.imshow(simulation_window(), aspect='auto', vmin=0, vmax=1)
    fig.colorbar(updater, ax=ax)
    def updatefig(*args):
        v = simulation_window()
        updater.set_array(v[::-1,:]) # Change the order of the units
        return updater,
    animation.FuncAnimation(fig, updatefig, interval=200, blit=True)
    plt.show()
