import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ODESolver import RungeKutta4
from PlateletsModel import Thrombopoiesis
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def event_func_150(c): # event when platelet count reaches 150/nl
    return np.any(c[6]>150*5e9/70)


####----MAIN----####
plt.style.use('seaborn-v0_8-poster')
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

# Record the recovery time with different initial conditions:
# initialize numpy arrays to store recovery time
recovery_HSC = np.zeros(9)
recovery_MPP = np.zeros(9)
recovery_CMP = np.zeros(9)
recovery_MEP = np.zeros(9)
recovery_all = np.zeros(9)
cd34_manual = np.zeros(9)
# load parameters:
params = config["params_avg"]
# simulation time
time = np.linspace(0, 200, 20000+1)
# loop through different initial conditions
index = 0
# alpha is the proportionality factor for each cell type:
for alpha in [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]:
    c_initial_1 = np.concatenate(
        (np.array([cd34 * 0.0408 * alpha, cd34 * 0.0719, cd34 * 0.2839, cd34 * 0.1484, 0, 0, 0]), c_start_plt),
        axis=None)
    c1, t1, te_1, ce1 = solve_ODE('Plt_7', c_initial_1, time, params, event_func_50)
    c_initial_2 = np.concatenate(
        (np.array([cd34 * 0.0408, cd34 * 0.0719 * alpha, cd34 * 0.2839 , cd34 * 0.1484, 0, 0, 0]), c_start_plt),
        axis=None)
    c2, t2, te_2, ce2 = solve_ODE('Plt_7', c_initial_2, time, params, event_func_50)
    c_initial_3 = np.concatenate(
        (np.array([cd34 * 0.0408, cd34 * 0.0719, cd34 * 0.2839 * alpha, cd34 * 0.1484, 0, 0, 0]), c_start_plt),
        axis=None)
    c3, t3, te_3, ce3 = solve_ODE('Plt_7', c_initial_3, time, params, event_func_50)
    c_initial_4 = np.concatenate(
        (np.array([cd34 * 0.0408, cd34 * 0.0719, cd34 * 0.2839, cd34 * 0.1484 * alpha, 0, 0, 0]), c_start_plt),
        axis=None)
    c4, t4, te_4, ce4 = solve_ODE('Plt_7', c_initial_4, time, params, event_func_50)
    c_initial_5 = np.concatenate(
        (np.array([cd34 * 0.0408 * alpha, cd34 * 0.0719 * alpha, cd34 * 0.2839 * alpha, cd34 * 0.1484 * alpha, 0, 0, 0]), c_start_plt),
        axis=None)
    c5, t5, te_5, ce5 = solve_ODE('Plt_7', c_initial_5, time, params, event_func_50)
    # record the recovery time:
    recovery_HSC[index] = te_1
    recovery_MPP[index] = te_2
    recovery_CMP[index] = te_3
    recovery_MEP[index] = te_4
    recovery_all[index] = te_5
    # record the proportionality factor:
    cd34_manual[index] = alpha
    # update the index:
    index = index+1

# Normalize the percentage of each cell type:
# for alpha in [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]:
#     c_initial_1 = np.concatenate(
#         (np.array([cd34 * 0.1 * alpha, cd34 * 0.1, cd34 * 0.1, cd34 * 0.1, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c1, t1, te_1, ce1 = solve_ODE('Plt_7', c_initial_1, time, params, event_func_50)
#     c_initial_2 = np.concatenate(
#         (np.array([cd34 * 0.1, cd34 * 0.1 * alpha, cd34 * 0.1 , cd34 * 0.1, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c2, t2, te_2, ce2 = solve_ODE('Plt_7', c_initial_2, time, params, event_func_50)
#     c_initial_3 = np.concatenate(
#         (np.array([cd34 * 0.1, cd34 * 0.1, cd34 * 0.1 * alpha, cd34 * 0.1, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c3, t3, te_3, ce3 = solve_ODE('Plt_7', c_initial_3, time, params, event_func_50)
#     c_initial_4 = np.concatenate(
#         (np.array([cd34 * 0.1, cd34 * 0.1, cd34 * 0.1, cd34 * 0.1 * alpha, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c4, t4, te_4, ce4 = solve_ODE('Plt_7', c_initial_4, time, params, event_func_50)
#     c_initial_5 = np.concatenate(
#         (np.array([cd34 * 0.1 * alpha, cd34 * 0.1 * alpha, cd34 * 0.1 * alpha, cd34 * 0.1 * alpha, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c5, t5, te_5, ce5 = solve_ODE('Plt_7', c_initial_5, time, params, event_func_50)
#     recovery_HSC[index] = te_1
#     recovery_MPP[index] = te_2
#     recovery_CMP[index] = te_3
#     recovery_MEP[index] = te_4
#     recovery_all[index] = te_5
#     cd34_manual[index] = alpha
#     index = index+1

# Plot the recovery time with different initial conditions:
plt.figure(figsize=(10,6))
plt.plot(cd34_manual, recovery_HSC, '-o',color='r', label='HSC')
plt.plot(cd34_manual, recovery_MPP, '-o',color='b', label='MPP')
plt.plot(cd34_manual, recovery_CMP, '-o',color='g', label='CMP')
plt.plot(cd34_manual, recovery_MEP, '-o',color='y', label='MEP')
plt.plot(cd34_manual, recovery_all, '-o',color='k', label='All')
plt.xscale('log')
plt.legend()
plt.show()