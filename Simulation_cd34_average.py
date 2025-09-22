import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ODESolver import RungeKutta4
from PlateletsModel import Thrombopoiesis
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function: load config data
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

# event functions for ode simulation, cease simulation when event occurs:
def event_func_20(c): # event when platelet count reaches 20/nl
    return np.any(c[6]>20*5e9/70)

def event_func_50(c): # event when platelet count reaches 50/nl
    return np.any(c[6]>50*5e9/70)


####----MAIN----####
# Set plot style
plt.style.use('seaborn-poster')
# load config data
config_path = "config/params.yaml"
config = load_config(config_path)
# load data from csv
df_average = pd.read_csv("data/data_average.csv")
# initial conditions
c_start_plt = df_average[df_average.iloc[:,0]==0]['Average']* 5e9 / 70
cd34 = 3.47e6 # average transfused CD34+ cell count calculated from clinical data
# Calculated the transfused progenitor components based on reference:
SUM = cd34 * (0.04048 + 0.0719 + 0.2839 + 0.1484)
HSC = cd34 * 0.0408
MPP = cd34 * 0.0719
CMP = cd34 * 0.2839
MEP = cd34 * 0.1484
# Set simulation parameters
params = config["params_avg"]
# time span for simulation
time = np.linspace(0, 200, 20000+1)
# Test model with different cd34 input
# Scenario 1: increase or decrease (times alpha) the absolute number of certain type of cells in CD34+ doses without normalization:
# Initialize index for storing results
index = 0
# Initialize arrays to store recovery times
recovery_20_HSC2 = np.zeros(22)
recovery_50_HSC2 = np.zeros(22)
recovery_20_MPP2 = np.zeros(22)
recovery_50_MPP2 = np.zeros(22)
recovery_20_CMP2 = np.zeros(22)
recovery_50_CMP2 = np.zeros(22)
recovery_20_MEP2 = np.zeros(22)
recovery_50_MEP2 = np.zeros(22)
recovery_20_all = np.zeros(22)
recovery_50_all = np.zeros(22)
# Initialize array to store fold change values
cd34_manual = np.zeros(22)
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    c_initial_1 = np.concatenate(
        (np.array([HSC * alpha, MPP, CMP, MEP, 0, 0, 0]), c_start_plt),
        axis=None)
    c1, t1, ce1, te_13 = solve_ODE('Plt_7', c_initial_1, time, params, event_func_20)
    c1, t1, ce1, te_14 = solve_ODE('Plt_7', c_initial_1, time, params, event_func_50)
    c_initial_2 = np.concatenate(
        (np.array([HSC, MPP * alpha, CMP, MEP, 0, 0, 0]), c_start_plt),
        axis=None)
    c2, t2, ce2, te_23 = solve_ODE('Plt_7', c_initial_2, time, params, event_func_20)
    c2, t2, ce2, te_24 = solve_ODE('Plt_7', c_initial_2, time, params, event_func_50)
    c_initial_3 = np.concatenate(
        (np.array([HSC, MPP, CMP * alpha, MEP, 0, 0, 0]), c_start_plt),
        axis=None)
    c3, t3, ce3, te_33 = solve_ODE('Plt_7', c_initial_3, time, params, event_func_20)
    c3, t3, ce3, te_34 = solve_ODE('Plt_7', c_initial_3, time, params, event_func_50)
    c_initial_4 = np.concatenate(
        (np.array([HSC, MPP, CMP, MEP * alpha, 0, 0, 0]), c_start_plt),
        axis=None)
    c4, t4, ce4, te_43 = solve_ODE('Plt_7', c_initial_4, time, params, event_func_20)
    c4, t4, ce4, te_44 = solve_ODE('Plt_7', c_initial_4, time, params, event_func_50)
    c_initial_5 = np.concatenate(
        (np.array([HSC * alpha, MPP * alpha, CMP * alpha, MEP * alpha, 0, 0, 0]), c_start_plt),
        axis=None)
    c5, t5, ce5, te_53 = solve_ODE('Plt_7', c_initial_5, time, params, event_func_20)
    c5, t5, ce5, te_54= solve_ODE('Plt_7', c_initial_5, time, params, event_func_50)
    recovery_20_HSC2[index] = te_13
    recovery_50_HSC2[index] = te_14
    recovery_20_MPP2[index] = te_23
    recovery_50_MPP2[index] = te_24
    recovery_20_CMP2[index] = te_33
    recovery_50_CMP2[index] = te_34
    recovery_20_MEP2[index] = te_43
    recovery_50_MEP2[index] = te_44
    recovery_20_all[index] = te_53
    recovery_50_all[index] = te_54
    cd34_manual[index] = alpha
    index = index+1

plt.figure(figsize=(10,6))
plt.plot(cd34_manual, recovery_20_HSC2, '-o',color='r', label='HSC')
plt.plot(cd34_manual, recovery_20_MPP2, '-o',color='b', label='MPP')
plt.plot(cd34_manual, recovery_20_CMP2, '-o',color='g', label='CMP')
plt.plot(cd34_manual, recovery_20_MEP2, '-o',color='y', label='MEP')
plt.plot(cd34_manual, recovery_20_all, '-o',color='k', label='all')
plt.legend()
plt.xlabel("Fold change of abundance")
plt.ylabel("Time to achieve 20/nl (days)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(cd34_manual, recovery_50_HSC2, '-o',color='r', label='HSC')
plt.plot(cd34_manual, recovery_50_MPP2, '-o',color='b', label='MPP')
plt.plot(cd34_manual, recovery_50_CMP2, '-o',color='g', label='CMP')
plt.plot(cd34_manual, recovery_50_MEP2, '-o',color='y', label='MEP')
plt.plot(cd34_manual, recovery_50_all, '-o',color='k', label='all')
plt.legend()
plt.xlabel("Fold change of abundance")
plt.ylabel("Time to achieve 50/nl (days)")
plt.tight_layout()
plt.show()

# Scenario 2: increase or decrease (times alpha) the relative proportion of certain type of cells in CD34+ doses with normalization:
# Initialize  index for storing results
index = 0
# Initialize arrays to store recovery times
recovery_20_HSC = np.zeros(22)
recovery_50_HSC = np.zeros(22)
recovery_20_MPP = np.zeros(22)
recovery_50_MPP = np.zeros(22)
recovery_20_CMP = np.zeros(22)
recovery_50_CMP = np.zeros(22)
recovery_20_MEP = np.zeros(22)
recovery_50_MEP = np.zeros(22)

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    c_initial_1 = np.concatenate(
        (np.array([HSC * alpha, MPP, CMP, MEP, 0, 0, 0])*(SUM/(HSC * alpha+MPP+CMP+MEP)), c_start_plt),
        axis=None)
    c1, t1, ce1, te_11 = solve_ODE('Plt_7', c_initial_1, time, params, event_func_20)
    c1, t1, ce1, te_12 = solve_ODE('Plt_7', c_initial_1, time, params, event_func_50)
    c_initial_2 = np.concatenate(
        (np.array([HSC, MPP * alpha, CMP, MEP, 0, 0, 0])*(SUM/(HSC+MPP * alpha+CMP+MEP)), c_start_plt),
        axis=None)
    c2, t2, ce2, te_21 = solve_ODE('Plt_7', c_initial_2, time, params, event_func_20)
    c2, t2, ce2, te_22 = solve_ODE('Plt_7', c_initial_2, time, params, event_func_50)
    c_initial_3 = np.concatenate(
        (np.array([HSC, MPP, CMP * alpha, MEP, 0, 0, 0])*(SUM/(HSC+MPP+CMP * alpha+MEP)), c_start_plt),
        axis=None)
    c3, t3, ce3, te_31 = solve_ODE('Plt_7', c_initial_3, time, params, event_func_20)
    c3, t3, ce3, te_32 = solve_ODE('Plt_7', c_initial_3, time, params, event_func_50)
    c_initial_4 = np.concatenate(
        (np.array([HSC, MPP, CMP, MEP * alpha, 0, 0, 0])*(SUM/(HSC+MPP+CMP+MEP * alpha)), c_start_plt),
        axis=None)
    c4, t4, ce4, te_41 = solve_ODE('Plt_7', c_initial_4, time, params, event_func_20)
    c4, t4, ce4, te_42 = solve_ODE('Plt_7', c_initial_4, time, params, event_func_50)

    recovery_20_HSC[index] = te_11
    recovery_50_HSC[index] = te_12
    recovery_20_MPP[index] = te_21
    recovery_50_MPP[index] = te_22
    recovery_20_CMP[index] = te_31
    recovery_50_CMP[index] = te_32
    recovery_20_MEP[index] = te_41
    recovery_50_MEP[index] = te_42
    index = index+1

plt.figure(figsize=(10,6))
plt.plot(cd34_manual, recovery_20_HSC, '-o',color='r', label='HSC')
plt.plot(cd34_manual, recovery_20_MPP, '-o',color='b', label='MPP')
plt.plot(cd34_manual, recovery_20_CMP, '-o',color='g', label='CMP')
plt.plot(cd34_manual, recovery_20_MEP, '-o',color='y', label='MEP')
plt.legend()
plt.xlabel("Fold change of abundance")
plt.ylabel("Time to achieve 20/nl (days)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(cd34_manual, recovery_50_HSC, '-o',color='r', label='HSC')
plt.plot(cd34_manual, recovery_50_MPP, '-o',color='b', label='MPP')
plt.plot(cd34_manual, recovery_50_CMP, '-o',color='g', label='CMP')
plt.plot(cd34_manual, recovery_50_MEP, '-o',color='y', label='MEP')
plt.legend()
plt.xlabel("Fold change of abundance")
plt.ylabel("Time to achieve 50/nl (days)")
plt.tight_layout()
plt.show()
