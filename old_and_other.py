import numpy as np
import matplotlib.pyplot as plt
from build_model import Unit

def check_units():
    """Create a binary, leaky and leaky onset unit to check whether they behave as expected
    Turn the binary unit on and off and plot the firing rates"""

    units = {}
    units["Binary"] = Unit("binary"); units["Binary"].switch_activation_binary_units()
    units["Leaky"] = Unit("leaky"); units["Dopaminergic"] = Unit("dopaminergic")
    units["Leaky Onset"] = Unit("leaky onset", tau=500, tau_i=500)
    units["Leaky"].add_connections([[units["Binary"], 1]]); units["Dopaminergic"].add_connections([[units["Binary"], 1]])
    units["Leaky Onset"].add_connections([[units["Binary"], 1]])

    t = np.arange(0, 10000, 1)
    for i_t in range(1, len(t)):
        if i_t == len(t) / 2:  # Switch food off again
            units["Binary"].switch_activation_binary_units()
        for i_u, unit in enumerate(units.values()):
            unit.activity(1)

    plt.figure()
    subplots = np.arange(len(units)*3).reshape(len(units),3) + 1
    titles = ["Potential","Firing rate", "Trace"]
    for i_u, (name, unit) in enumerate(units.items()):
        activity = np.array(unit.activity_history)
        for i,title in enumerate(titles):
            plt.subplot(len(units), 3, subplots[i_u,i]); plt.plot(activity[:,i])
            plt.title(name, fontsize=7)
            plt.ylabel(title, fontsize=5)
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
    plt.subplots_adjust(hspace=1)
    plt.show()

check_units()

# Test with only one stimulus if the lever gets associated with the food (weights update as expected)
def test_instrumental_training():

    # Define the units needed
    units = {}
    units["Lever"] = Unit("binary")
    units["Food"] = Unit("binary")
    units["CS"] = Unit("leaky onset", tau=500, tau_i=500)
    units["US"] = Unit("leaky onset", tau=500, tau_i=500)
    units["LH"] = Unit("leaky onset", tau=100, tau_i=500)
    units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=4)
    # Connect them
    units["CS"].add_connections([[units["Lever"], 5], [units["US"], 0], [units["VTA"], 1]])
    units["US"].add_connections([[units["Food"], 5], [units["CS"], 0], [units["VTA"], 1]]) # Why deosn't it work that way!
    units["LH"].add_connections([[units["Food"], 10], [units["US"], 5]])
    units["VTA"].add_connections([[units["LH"], 20]])

    # Run the model
    t = np.arange(0, 1000000, 1)
    lever_on_switches = np.linspace(0,len(t), 10,dtype=int)
    food_on_switches = lever_on_switches + 5000
    for i_t in range(len(t)):
        if i_t in lever_on_switches:  # Switch food off again
            units["Lever"].switch_activation_binary_units()
        elif i_t in lever_on_switches+10000:
            units["Lever"].switch_activation_binary_units()
        if i_t in food_on_switches:  # Switch food off again
            units["Food"].switch_activation_binary_units()
        elif i_t in food_on_switches + 10000:
            units["Food"].switch_activation_binary_units()
        for i_u, unit in enumerate(units.values()):
            unit.activity(1)

    # Plot the behaviour
    plt.figure()
    subplots = np.arange(len(units) * 3).reshape(len(units), 3) + 1
    titles = ["Potential", "Firing rate", "Trace"]
    for i_u, (name, unit) in enumerate(units.items()):
        activity = np.array(unit.activity_history)
        for i, title in enumerate(titles):
            plt.subplot(len(units), 3, subplots[i_u, i])
            plt.plot(activity[:, i])
            plt.title(name, fontsize=7)
            plt.ylabel(title, fontsize=5)
            plt.yticks(fontsize=5)
            plt.xticks(fontsize=5)
    plt.subplots_adjust(hspace=1)

    # Plot th weights from CS(Lever) to US(Food) (should increase)
    plt.figure()
    weights = units["US"].connections[1].weight_history
    plt.plot(weights)
    plt.title("Weights from CS to US")
    plt.show()
    print("debug")

# Run the functions of interest
test_instrumental_training()
check_units()


###########################################################################################################
# Model with fixed weights (maximum weights- no learning)
def build_fixed_model():

    # Initialize the units of the model (start with the goal loop)
    units = {}
    units["Lever"] = Unit("binary"); units["Chain"] = Unit("binary")
    units["FoodA"] = Unit("binary"); units["FoodB"] = Unit("binary")
    units["SatA"] = Unit("binary"); units["SatB"] = Unit("binary")
    units["CSa"] = Unit("leaky onset", tau=500, tau_i=500); units["CSb"] = Unit("leaky onset", tau=500, tau_i=500)
    units["USa"] = Unit("leaky onset", tau=500, tau_i=500); units["USb"] = Unit("leaky onset", tau=500, tau_i=500)
    units["LH"] = Unit("leaky onset", tau=100, tau_i=500)
    units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=1.5)
    units["NAc_1"] = Unit("leaky"); units["NAc_2"] = Unit("leaky")
    units["STNv_1"] = Unit("leaky"); units["STNv_2"] = Unit("leaky")
    units["SNpr_1"] = Unit("leaky"); units["SNpr_2"] = Unit("leaky")
    units["DM_1"] = Unit("leaky", noise_coeff=6); units["DM_2"] = Unit("leaky", noise_coeff=6)
    units["PL_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.8); units["PL_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.8)

    # Connect the units
    # BLA units connected to themselves?
    units["CSa"].add_connections([[units["Lever"], 5], [units["CSb"], 2], [units["USa"], 2], [units["USb"], 2], [units["VTA"], 1]])
    units["CSb"].add_connections([[units["Chain"], 5], [units["CSa"], 2], [units["USa"], 2], [units["USb"], 2], [units["VTA"], 1]])
    units["USa"].add_connections([[units["FoodA"], 5], [units["FoodB"], 5], [units["SatA"], -10], [units["SatB"], -10], [units["CSa"], 2], [units["CSb"], 2], [units["USb"],2], [units["VTA"], 1]])
    units["USb"].add_connections([[units["FoodA"], 5], [units["FoodB"], 5], [units["SatA"], -10], [units["SatB"], -10],[units["CSa"], 2], [units["CSb"], 2], [units["USa"], 2], [units["VTA"], 1]])
    units["LH"].add_connections([[units["FoodA"], 10], [units["FoodB"], 10], [units["USa"], 5], [units["USb"], 5]])
    units["VTA"].add_connections([[units["LH"], 20]])
    units["NAc_1"].add_connections([[units["VTA"], 1], [units["USa"], 2], [units["USb"], 2], [units["PL_1"], 1]])
    units["NAc_2"].add_connections([[units["VTA"], 1], [units["USa"], 2], [units["USb"], 2], [units["PL_2"], 1]])
    units["STNv_1"].add_connections([[units["PL_1"], 1.6]])
    units["STNv_2"].add_connections([[units["PL_2"], 1.6]])
    units["SNpr_1"].add_connections([[units["NAc_1"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["SNpr_2"].add_connections([[units["NAc_2"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["DM_1"].add_connections([[units["SNpr_1"], -1.5], [units["DM_1"], 1], [units["DM_2"], -0.8]])
    units["DM_2"].add_connections([[units["SNpr_2"], -1.5],  [units["DM_1"], -0.8], [units["DM_2"], 1]])
    units["PL_1"].add_connections([[units["DM_1"], 1]])
    units["PL_2"].add_connections([[units["DM_2"], 1]])

    # Add the names to the class objects
    for name, unit in units.items():
        unit.name = name

    return units

###########################################################################################################
# Build a small model for testing
def build_test_model():
    units = {}
    units["Binary"] = Unit("binary")
    units["Leaky"] = Unit("leaky")
    units["Dopaminergic"] = Unit("dopaminergic")
    units["Leaky Onset"] = Unit("leaky onset", tau=500, tau_i=500)
    units["Leaky"].add_connections([[units["Binary"], 1]])
    units["Dopaminergic"].add_connections([[units["Binary"], 1]])
    units["Leaky Onset"].add_connections([[units["Binary"], 1]])

    # Add the names to the class objects
    for name, unit in units.items():
        unit.name = name

    return units

# Build a model of the BLA with only one lever and food
def build_BLA_model():
    # Define the units needed
    units = {}
    units["Lever"] = Unit("binary")
    units["Food"] = Unit("binary")
    units["CS"] = Unit("leaky onset", tau=500, tau_i=500)
    units["US"] = Unit("leaky onset", tau=500, tau_i=500)
    units["LH"] = Unit("leaky onset", tau=100, tau_i=500)
    units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=4)
    # Connect them
    units["CS"].add_connections([[units["Lever"], 5], [units["US"], 0, "plastic"], [units["VTA"], 1]])
    units["US"].add_connections([[units["Food"], 5], [units["CS"], 0, "plastic"], [units["VTA"], 1]])
    units["LH"].add_connections([[units["Food"], 10], [units["US"], 5]])
    units["VTA"].add_connections([[units["LH"], 20]])

    # Add the names to the class objects
    for name, unit in units.items():
        unit.name = name
    return units

###########################################################################################################
def build_fixed_action_loop():
    # Initialize the units of the model, leave out the amygdala for now
    units = {}
    units["Cue"] = Unit("binary")
    units["Amgydala"] = Unit("binary")
    units["DLS_1"] = Unit("leaky")
    units["DLS_2"] = Unit("leaky")
    units["STNdl_1"] = Unit("leaky")
    units["STNdl_2"] = Unit("leaky")
    units["GPi_1"] = Unit("leaky")
    units["GPi_2"] = Unit("leaky")
    units["MGV_1"] = Unit("leaky", noise_coeff=6)
    units["MGV_2"] = Unit("leaky", noise_coeff=6)
    units["MC_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.8)
    units["MC_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.8)
    # Connect the units
    units["DLS_1"].add_connections([[units["Amgydala"], -1],[units["Cue"], 1], [units["MC_1"], 1]])
    units["DLS_2"].add_connections([[units["Amgydala"], 1], [units["MC_2"], 1]])
    units["STNdl_1"].add_connections([[units["MC_1"], 1.6]])
    units["STNdl_2"].add_connections([[units["MC_2"], 1.6]])
    units["GPi_1"].add_connections([[units["DLS_1"], -3], [units["STNdl_1"], 2], [units["STNdl_2"], 2]])
    units["GPi_2"].add_connections([[units["DLS_2"], -3], [units["STNdl_1"], 2], [units["STNdl_2"], 2]])
    units["MGV_1"].add_connections([[units["GPi_1"], -1.5], [units["MGV_1"], 1], [units["MGV_2"], -0.8]])
    units["MGV_2"].add_connections([[units["GPi_2"], -1.5], [units["MGV_1"], -0.8], [units["MGV_2"], 1]])
    units["MC_1"].add_connections([[units["MGV_1"], 1]])
    units["MC_2"].add_connections([[units["MGV_2"], 1]])

    # Add the names to the class objects
    for name, unit in units.items():
        unit.name = name
    return units


#Define the times at which the binary units should be switched
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

def build_goal_loop():

    # Initialize the units of the model (start with the goal loop)
    units = {}
    units["Cue"] = Unit("binary"); units["Amygdala"] = Unit("binary")
    units["VTA"] = Unit("dopaminergic", dopa_in=0.8, dopa_de=4)
    units["NAc_1"] = Unit("leaky"); units["NAc_2"] = Unit("leaky")
    units["STNv_1"] = Unit("leaky"); units["STNv_2"] = Unit("leaky")
    units["SNpr_1"] = Unit("leaky"); units["SNpr_2"] = Unit("leaky")
    units["DM_1"] = Unit("leaky", noise_coeff=6); units["DM_2"] = Unit("leaky", noise_coeff=6)
    units["PL_1"] = Unit("leaky", tau=2000, sigma=20, thres=0.8); units["PL_2"] = Unit("leaky", tau=2000, sigma=20, thres=0.8)

    # Connect the units - fixed at maximum connection weight
    units["VTA"].add_connections([[units["Amygdala"], 2]])
    # What connection weight between teh cue and the NAc? In the MM the maximum connection weight was 2 for each and
    # each US was connected to each NAc --> Hence a connection from the cue to both NAc with max weight 4?
    units["NAc_1"].add_connections([[units["Cue"], 2], [units["PL_1"], 1]])
    units["NAc_2"].add_connections([[units["Cue"], 0.5], [units["VTA"], 1], [units["PL_2"], 1]])
    units["STNv_1"].add_connections([[units["PL_1"], 1]])
    units["STNv_2"].add_connections([[units["PL_2"], 1]])
    units["SNpr_1"].add_connections([[units["NAc_1"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    units["SNpr_2"].add_connections([[units["NAc_2"], -3], [units["STNv_1"], 2], [units["STNv_2"], 2]])
    # What is the weight of the connection of a DM unit on itself? Chose default: 1
    units["DM_1"].add_connections([[units["SNpr_1"], -1.5], [units["DM_2"], -0.8], [units["DM_1"], 1]])
    units["DM_2"].add_connections([[units["SNpr_2"], -1.5], [units["DM_1"], -0.8], [units["DM_2"], 1]])
    units["PL_1"].add_connections([[units["DM_1"], 1]])
    units["PL_2"].add_connections([[units["DM_2"], 1]])

    # Add the names to the class objects
    for name,unit in units.items():
        unit.name = name
    return units

    #######################################################################################################
    # Get the weight of the input (dependent on dopaminergic input units)
    def get_input_weight(self):
        # Check if the unit is connected to a dopaminergic unit
        dopa_input_unit = self.get_dopa_input_unit()
        if dopa_input_unit:
            input_weight = dopa_input_unit.dopa_in + dopa_input_unit.dopa_de * dopa_input_unit.firing_rate
        else:
            input_weight = 1
        return input_weight

    #######################################################################################################
    # Return the dopaminergic input unit if there is one, else none
    def get_dopa_input_unit(self):
        dopa_input_unit = [connection.input_unit for connection in self.connections
                           if connection.input_unit.type == "dopaminergic"]
        if dopa_input_unit:
            return dopa_input_unit[0]
        else:
            return None


        # Update the weights - only for units that receive dopaminergic input
        dopa_input_unit = self.get_dopa_input_unit()
        if dopa_input_unit:
            for connection in self.connections:
                # Update only plastic connections
                if connection.type == "plastic":
                    # Leaky onset units (BLA learning)
                    if self.type == "leaky onset":
                        weight_change = self.eta_bla * pos_sat(dopa_input_unit.firing_rate - self.thres_da_bla) * \
                                        pos_sat(self.trace_change) * neg_sat(connection.input_unit.trace_change) * \
                                        (self.max_w_bla - connection.weight)
                    # Leaky units (Striatal learning)
                    elif self.type == "leaky":
                        weight_change = self.eta_str * pos_sat(dopa_input_unit.firing_rate - self.thres_da_str) * \
                                        pos_sat(self.firing_rate - self.thres_str) *\
                                        pos_sat(connection.input_unit.firing_rate  - self.thres_inp_str)
                    connection.weight = connection.weight + weight_change * dt
                    # Save the connection weight in an array
                    connection.weight_history.append(connection.weight)