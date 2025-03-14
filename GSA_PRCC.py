# Global sensitivity analysis using the Partial Rank Correlation Coefficient (PRCC) method

import numpy as np
from ODESolver import RungeKutta4
from PlateletsModel import Thrombopoiesis
from matplotlib import pyplot as plt
from scipy.stats import qmc, spearmanr

def solve_ODE(ODEmodel, initial, time, params, event_func=None):
    model = Thrombopoiesis(parameters=params)
    solver = RungeKutta4(getattr(model, ODEmodel), event_func=event_func)
    solver.set_initial_condition(initial)
    c, t, ce, te = solver.solve(time)
    return [c, t, ce, te]

# free parameters for estimate and output cell types:
para_label=['p1','p2','p3','p4','e5','e6','a1','a2','a3','a4','kp','d7','k_shed','dummy']
output_label=['c1(HSC)','c2(MPP)','c3(CMP)','c4(MEP)','c5(Mk-blast)','c6(Mk)','c7(Platelets)']
# lower and upper bounds for the parameters
lb = np.asarray([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1])  # lower bound
ub = np.asarray([0.05,  0.2, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 15,  0.2, 5.0, 10])  # upper bound
# number of parameters + 1 dummy parameter
num_params = 13+1
# number of sampling
runs = 1000
sampler = qmc.LatinHypercube(d=num_params)
sample = sampler.random(n=runs)
p_LHS = qmc.scale(sample, lb, ub)

# Initial conditions
cd34 = 3.5e6
plt_start = 197*5e9/70
c_initial = np.array([cd34*0.0408, cd34*0.0719, cd34*0.2839, cd34*0.1484, 0, 0, 0, plt_start])
time = np.linspace(0, 500, 1001) # 500 days simulation time
timepoints = np.array([20,800]) # We are interested in the platelet count at 10 (increasing phase) and 400 (plateau)days
# initialize the matrix to store the results
C_LHS = np.zeros((runs, len(time), 7))
# solve the ODE model for each parameter sampling group:
for i in range(runs):
    print(i)
    para_i = {
        "p_c": p_LHS[i,0:6],
        "a_c": p_LHS[i,6:10],
        "k_p": p_LHS[i,10]*1e-10,
        "k_a": (2*p_LHS[i,6]-1)/1.4e10,
        "d_plt": p_LHS[i,11],
        "k_decline": 0.345,
        "k_shed": p_LHS[i,12]*1000,
    }
    c, t, ce, te = solve_ODE('Plt_7',c_initial,time,para_i)
    C_LHS[i] = c[:,0:7]

# Calculate the PRCC for each parameter at the increasing phase (t0) and plateau phase (t1) of the platelet count
# change the target to check sensitivity to other cell types
# 0: HSC, 1: MPP, 2: CMP, 3: MEP, 4: Mk-blast, 5: Mk, 6: Platelets
target_output = 6
# Initialize the matrix to store the PRCC values
prcc_increase = np.zeros(num_params)
prcc_plateau = np.zeros(num_params)
for i in range(num_params):
    prcc_increase[i], _ = spearmanr(p_LHS[:, i], C_LHS[:, timepoints[0], target_output])
    prcc_plateau[i], _ = spearmanr(p_LHS[:, i], C_LHS[:, timepoints[1], target_output])

# Plot the PRCC values:
plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.bar(para_label,prcc_increase)
plt.title("plt_t0")
plt.subplot(1,2,2)
plt.bar(para_label,prcc_plateau)
plt.title("plt_t1")
plt.tight_layout()
plt.show()