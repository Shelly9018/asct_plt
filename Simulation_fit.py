import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ODESolver import RungeKutta4
from PlateletsModel import Thrombopoiesis
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function: load configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function: solve ODE model
def solve_ODE(ODEmodel, initial, time, params, event_func=None):
    model = Thrombopoiesis(parameters=params)
    solver = RungeKutta4(getattr(model, ODEmodel), event_func=event_func)
    solver.set_initial_condition(initial)
    c, t, ce, te = solver.solve(time)
    return [c, t, ce, te]

# Function: plot immature cell types
def plot_MK(t, c):
    # Cell type labels:
    clabel = ['HSC', 'MPP','CMP','MEP', 'MKb', 'MK']
    # Create the figure and axes
    fig1, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(18, 12))
    # Flatten the axes array for easy iteration
    axes_flat = axes.flat

    # Loop through the axes and plot data
    for i, ax in enumerate(axes_flat):
        ax.plot(t, c[:, i])
        ax.set_title(clabel[i])
        ax.set_ylim(bottom=0)  # Set lower y-limit to 0

    # Set y-axis limits for specific plots (if needed)
    axes[0, 0].set_ylim(0, 5e5)
    axes[0, 1].set_ylim(0, 1e6)

    # Set x-label for bottom row
    axes[2, 0].set_xlabel('Time (days)')
    axes[2, 1].set_xlabel('Time (days)')

    # Set y-label for left column
    axes[0, 0].set_ylabel('Cells (1/kg)')
    axes[1, 0].set_ylabel('Cells (1/kg)')
    axes[2, 0].set_ylabel('Cells (1/kg)')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Function: plot average platelet data fitting results
def plot_average(t, c):
    plt.figure(figsize=(10,6))
    plt.scatter(np.asarray(df_average.iloc[:, 0]), np.asarray(df_average.iloc[:, 1]),color='r')
    plt.plot(t[0:200], (c[0:200, -2] + c[0:200, -1]) * 70 / 5e9, linewidth=3)
    plt.tight_layout()
    plt.show()

####----MAIN----####
plt.style.use('seaborn-poster')
# load config data
config_path = "config/params.yaml"
config = load_config(config_path)
# load data from csv
df_average = pd.read_csv("data/data_average.csv")
# initial conditions
c_start_plt = df_average[df_average.iloc[:,0]==0]['Average']* 5e9 / 70
cd34 = 3.47e6 # average transfused CD34+ cell count calculated from clinical data
# percentage of each cell type taken from reference (Scala et al. 2023)
c_initial = np.concatenate((np.array([cd34*0.0408, cd34*0.0719, cd34*0.2839, cd34*0.1484, 0, 0, 0]), c_start_plt), axis=None)
days = 1000 # simulation time
time = np.linspace(0, days, 2*days+1) # convert into numpy array
# run simulation with average parameter set
# config parameters can be changed to individual patient parameter sets
c, t, ce, te = solve_ODE('Plt_7', c_initial, time, config["params_avg"])
# plot the results
plot_average(t,c)
plot_MK(t,c)