import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ODESolver import RungeKutta4
import time
from PlateletsModel import Thrombopoiesis
import yaml
from scipy import stats
from scipy.stats import spearmanr
from heatmap import heatmap,corrplot
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def event_func_20(c):
    return np.any(c[6]>20*5e9/70)

def event_func_50(c):
    return np.any(c[6]>50*5e9/70)

def event_func_150(c):
    return np.any(c[6]>150*5e9/70)

def solve_ODE(ODEmodel, initial, time, params, event_func=None):
    model = Thrombopoiesis(parameters=params)
    solver = RungeKutta4(getattr(model, ODEmodel), event_func=event_func)
    solver.set_initial_condition(initial)
    c, t, te, ce = solver.solve(time)
    return [c, t, te, ce]

def plot_MK(t, c, n):
    label = ['HSC', 'MPP','CMP','MEP', 'MKb', 'MK']
    fig1, axs1 = plt.subplots(nrows=n, ncols=1, sharex='all', figsize=(18, 12))
    for i in range(n):
        axs1[i].plot(t, c[:, i], linewidth=3, label=label[i])
        axs1[i].legend()
    axs1[0].axis(ymin=0, ymax=5e5)
    axs1[1].axis(ymin=0, ymax=1e6)
    axs1[n-2].set_xlabel('Time (days)')
    axs1[(n-1)//2].set_ylabel('Cells (1/kg)')
    plt.tight_layout()
    fig1.show()

def plot_meanvalue(t, c):
    plt.figure(figsize=(10,6))
    for i in range(53):
        plt.plot(np.asarray(Days.iloc[:, i]), np.asarray(PatientsData.iloc[:, i]), '-o', color='lightgrey')
    plt.plot(np.asarray(df_patients.iloc[:, 0]), np.asarray(df_meanvalue), '-o',color='r')
    plt.plot(t[0:200], (c[0:200, -2] + c[0:200, -1]) * 70 / 5e9, linewidth=3)
    plt.tight_layout()
    plt.show()

def plot_fitting(t, c, id):
    plt.figure(figsize=(10,6))
    # plot the only patient data after day 0
    # plt.plot(np.asarray(Days.iloc[:, id-1]), np.asarray(PatientsData.iloc[:, id-1]) * 5e9 / 70, '-o', color='r')
    for i in range(53):
        plt.plot(np.asarray(Days.iloc[:, i]), np.asarray(PatientsData.iloc[:, i]), '-o', color='lightgrey')
    #plt.plot(np.asarray(df_patients.iloc[:, 0]), np.asarray(df_meanvalue), '-o', color='r')
    plt.plot(np.asarray(Days0), np.asarray(Plts0), '-o', color='green')

    plt.plot(t[0:200], (c[0:200, -2] + c[0:200, -1])*70/5e9, linewidth=3)
    #plt.title(f'{lab_time}')
    plt.xlim(0,100)
    plt.ylim(0, 500)
    plt.tight_layout()
    # plt.savefig(f'C:/Users/chzhu/Documents/BMT/Results/Patients_Background/Fitting_{id}.png')

    plt.show()

#def equilibrium_state(c, parameters):
def equilibrium(params):
    p_c = params['p_c']
    a_c = params['a_c']
    k_p = params['k_p']
    k_a = params['k_a']
    d_platelets = params['d_plt']
    k_frac = params['k_shed']
    # equilibrium state:
    c = np.zeros(7)
    c[6] = (2 * a_c[0] - 1) / k_a
    c[5] = (d_platelets * c[6]) / (k_frac * p_c[5])
    c[4] = (d_platelets * c[6]) / (k_frac * p_c[4])
    c[3] = (d_platelets * c[6] * (1 + k_p * c[6]) * a_c[0]) / (k_frac * (2 * a_c[0] - a_c[3]) * p_c[3])
    c[2] = (d_platelets * c[6] * (1 + k_p * c[6]) * (a_c[0] - a_c[3]) * a_c[0]) / (
                k_frac * (2 * a_c[0] - a_c[3]) * (2 * a_c[0] - a_c[2]) * p_c[2])
    c[1] = (d_platelets * c[6] * (1 + k_p * c[6]) * (a_c[0] - a_c[2]) * (a_c[0] - a_c[3]) * a_c[0]) / (
                k_frac * (2 * a_c[0] - a_c[3]) * (2 * a_c[0] - a_c[2]) * (2 * a_c[0] - a_c[1]) * p_c[1])
    c[0] = (d_platelets * c[6] * (1 + k_p * c[6]) * (a_c[0] - a_c[1]) * (a_c[0] - a_c[2]) * (a_c[0] - a_c[3])) / (
                k_frac * (2 * a_c[0] - a_c[3]) * (2 * a_c[0] - a_c[2]) * (2 * a_c[0] - a_c[1]) * p_c[0])
    return c

# Record the time of simulation
lab_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
plt.style.use('seaborn-poster')
# plt.style.use('seaborn-v0_8-poster')
####----MAIN----####
# load config data
config_path = "config/MK_7_config.yaml"
config = load_config(config_path)

# load data from multiple myeloma patients
df_plt_all = pd.read_csv(r'e:\vs_code\bmt\data\platelets_all.csv')
df_cd34_all = pd.read_csv(r'e:\vs_code\bmt\data\cd34_all.csv')
df_patients = pd.read_csv(r'e:\vs_code\bmt\data\patients_index.csv')

# load data on laptop:
# df_plt_all = pd.read_csv(r'C:\Users\ChenxuZhu\Documents\BMT\Data\platelets_all.csv')
# df_cd34_all = pd.read_csv(r'C:\Users\ChenxuZhu\Documents\BMT\Data\cd34_all.csv')
# df_patients = pd.read_csv(r'C:\Users\ChenxuZhu\Documents\BMT\Data\patients_index.csv')

Days = df_plt_all.iloc[:, [i % 2 == 0 for i in range(len(df_plt_all.columns))]]
PatientsData = df_plt_all.iloc[:, [i % 2 == 1 for i in range(len(df_plt_all.columns))]]

# # Average Simulation:
# df_meanvalue = df_patients.iloc[:,-1]
# mean_after0 = df_meanvalue[df_patients.iloc[:,0]>=0]
# cd34 = df_cd34_all['transfused CD34+ cells'].mean()*1e6
# c_start_plt = mean_after0.iloc[0] * 5e9 / 70
# c_initial = np.concatenate((np.array([cd34*0.0408, cd34*0.0719, cd34*0.2839, cd34*0.1484, 0, 0, 0]), c_start_plt), axis=None)
# days = 1000
# time = np.linspace(0, days, 2*days+1)
# c, t, te, ce = solve_ODE('MK_7', c_initial, time, config["params_avg53"])

# # Individual Simulation:
# id = 18
# Days0 = Days.iloc[:, id-1]
# Plts0 = PatientsData.iloc[:, id-1]
# plot_meanvalue(t,c)
# plot_fitting(t,c,id)


# Test each individual patient
t_event = np.zeros([53,4])
results = np.zeros([53,9])
for id in range(1,54):
    if f'params_pat{id}' in config:
        filt = (Days.iloc[:, id-1] >= 0)
        Days0 = Days.iloc[filt[filt].index, id-1]
        Plts0 = PatientsData.iloc[filt[filt].index, id-1]
        cd34 = df_cd34_all['transfused CD34+ cells'].iloc[id-1]*1e6
        if config[f'params_pat{id}'].get('plt_start', False):  # Use .get() to avoid KeyError
            c_start_plt = float(config[f'params_pat{id}']['plt_start']) * 5e9 / 70
        else:
            c_start_plt = float(Plts0.iloc[0]) * 5e9 / 70

        c_initial = np.concatenate((np.array([cd34*0.0408, cd34*0.0719, cd34*0.2839, cd34*0.1484, 0, 0, 0]), c_start_plt), axis=None)
        days = 1000
        time = np.linspace(0, days, 2*days+1)
        c, t, te20, ce = solve_ODE('MK_7', c_initial, time, config[f"params_pat{id}"],event_func_20)
        c50, t50, te50, ce50 = solve_ODE('MK_7', c_initial, time, config[f"params_pat{id}"], event_func_50)
        #plot_MK(t, c, 6)
        #plot_fitting(t, c, id)
        t_event[id-1,0] = id
        t_event[id-1,1] = te20
        t_event[id - 1, 2] = te50
        t_event[id - 1, 3] = ce50[-1]*70/5e9
        params = config[f"params_pat{id}"]
        c_equ = equilibrium(params)
        s_a = 1 / (1 + c_equ[6] * params["k_a"])
        a_c_equ = np.array(
            [params['a_c'][0] * s_a, params['a_c'][1] * s_a, params['a_c'][2] * s_a, params['a_c'][3] * s_a])
        # Flux ratio for a2, a3, a:
        flux_ratio_1 = -2*(1-a_c_equ[1])/(2*a_c_equ[1]-1)
        flux_ratio_2 = -2*(1-a_c_equ[2])/(2*a_c_equ[2]-1)
        flux_ratio_3 = -2*(1-a_c_equ[3])/(2*a_c_equ[3]-1)
        # Convert proliferation rate to division time:
        s_p = 1/(1+ c_equ[6]*params["k_p"])
        p_c_equ = np.array(
            [params['p_c'][0] * s_p, params['p_c'][1] * s_p, params['p_c'][2] * s_p, params['p_c'][3] * s_p])
        p_day = np.log(2)/p_c_equ
        results[id-1,:] = np.concatenate((p_day, s_p,c_equ[-1]*70/5e9,flux_ratio_1,flux_ratio_2,flux_ratio_3), axis=None)
    else:
        continue

Plt_diagnosis = np.asarray(PatientsData.iloc[0, :])
index_pat = t_event[:,0] != 0
Plt_diagnosis_index = Plt_diagnosis[index_pat]
results_index = results[index_pat]

rho, p = spearmanr(Plt_diagnosis[[0,6,12,16,14,35,36,48]], results[[0,6,12,16,14,35,36,48],0])

# Group into high and low Plt
results_h4 = results[[0,6,12,16],:]
results_l4 = results[[14,35,36,48],:]
# t_event = t_event[index_pat]
# index_h = Plt_diagnosis >= 140
# index_l = Plt_diagnosis < 140
# t_event_h = t_event[index_h]
# results_h = results_index[index_h]
# t_event_l = t_event[index_l]
# results_l = results_index[index_l]

#t_stat, p_value = stats.ttest_ind(t_event_l[:,1]/results_l[:,-1], t_event_h[:,1]/results_h[:,-1],alternative='greater')
# 1. test cell counts at day 50:
# t_stat, p_value = stats.ttest_ind(t_event_l[:,3], t_event_h[:,3],alternative='less')
# 2. Correlation between cells eqiublirbium and time to recovery at 20nl:
# Matrix = np.zeros([2,45])
# A_matrix = np.concatenate((t_event_l[:,1],t_event_h[:,1]), axis=None)
# B_matrix = np.concatenate((results_l[:,5],results_h[:,5]), axis=None)
## Heatmap:
# Matrix[0,:] = A_matrix
# Matrix[1,:] = B_matrix
# rho = np.zeros((Matrix.shape[1]-1,Matrix.shape[1]-1))
# p = np.zeros((Matrix.shape[1]-1,Matrix.shape[1]-1))
# for i in range(1,Matrix.shape[1]):
#     for j in range(1,Matrix.shape[1]):
#         A = Matrix[:,i]
#         B = Matrix[:,j]
#         rho[i-1,j-1], p[i-1,j-1] = spearmanr(A, B)
# plt.rcParams['figure.figsize'] = (10,10)
# sns.set(color_codes=True, font_scale=1.2)
# plt.figure(figsize=(10,10))
# data_corr = pd.DataFrame(rho)
# corrplot(data_corr, size_scale=300)
# plt.show()


# Calculate the parameters value in equilirbium state:
# results = np.zeros((8, 5))
# index = 0
# for i in [1,7,13,17,15,36,37,49]:
#     params = config[f"params_pat{i}"]
#     c_equ = equilibrium(params)
#     s_p = 1/(1+ c_equ[6]*params["k_p"])
#     s_a = 1/(1+ c_equ[6]*params["k_a"])
#     p_c_equ = np.array([params['p_c'][0]*s_p, params['p_c'][1]*s_p, params['p_c'][2]*s_p, params['p_c'][3]*s_p])
#     a_c_equ = np.array([params['a_c'][0]*s_a, params['a_c'][1]*s_a, params['a_c'][2]*s_a, params['a_c'][3]*s_a])
#     p_day = np.log(2)/p_c_equ
#     d_day = np.log(2)/params['d_plt']
#     results[index,:] = np.concatenate((p_day, s_p), axis=None)
#     index += 1

# Test model with different cd34 input
# df_meanvalue = df_patients.iloc[:,-1]
# mean_after0 = df_meanvalue[df_patients.iloc[:,0]>=0]
# cd34 = df_cd34_all['transfused CD34+ cells'].mean()*1e6
# c_start_plt = mean_after0.iloc[0] * 5e9 / 70

# id = 49
# filt = (Days.iloc[:, id-1] >= 0)
# Days0 = Days.iloc[filt[filt].index, id-1]
# Plts0 = PatientsData.iloc[filt[filt].index, id-1]
# cd34 = df_cd34_all['transfused CD34+ cells'].iloc[id-1]*1e6
# #c_start_plt = float(config[f'params_pat{id}']['plt_start']) * 5e9 / 70
# c_start_plt = float(Plts0.iloc[0]) * 5e9 / 70
#
# recovery_20_HSC = np.zeros(10)
# recovery_20_MPP = np.zeros(10)
# recovery_20_CMP = np.zeros(10)
# recovery_20_MEP = np.zeros(10)
# recovery_20_all = np.zeros(10)
# cd34_manual = np.zeros(10)
# params = config[f"params_pat{id}"]
# time = np.linspace(0, 200, 20000+1)
# index = 0
# for alpha in [0.01, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]:
#     c_initial_1 = np.concatenate(
#         (np.array([cd34 * 0.0408 * alpha, cd34 * 0.0719, cd34 * 0.2839, cd34 * 0.1484, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c1, t1, te_1, ce1 = solve_ODE('MK_7', c_initial_1, time, params, event_func_150)
#     c_initial_2 = np.concatenate(
#         (np.array([cd34 * 0.0408, cd34 * 0.0719 * alpha, cd34 * 0.2839 , cd34 * 0.1484, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c2, t2, te_2, ce2 = solve_ODE('MK_7', c_initial_2, time, params, event_func_150)
#     c_initial_3 = np.concatenate(
#         (np.array([cd34 * 0.0408, cd34 * 0.0719, cd34 * 0.2839 * alpha, cd34 * 0.1484, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c3, t3, te_3, ce3 = solve_ODE('MK_7', c_initial_3, time, params, event_func_150)
#     c_initial_4 = np.concatenate(
#         (np.array([cd34 * 0.0408, cd34 * 0.0719, cd34 * 0.2839, cd34 * 0.1484 * alpha, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c4, t4, te_4, ce4 = solve_ODE('MK_7', c_initial_4, time, params, event_func_150)
#     c_initial_5 = np.concatenate(
#         (np.array([cd34 * 0.0408 * alpha, cd34 * 0.0719 * alpha, cd34 * 0.2839 * alpha, cd34 * 0.1484 * alpha, 0, 0, 0]), c_start_plt),
#         axis=None)
#     c5, t5, te_5, ce5 = solve_ODE('MK_7', c_initial_5, time, params, event_func_150)
#     recovery_20_HSC[index] = te_1
#     recovery_20_MPP[index] = te_2
#     recovery_20_CMP[index] = te_3
#     recovery_20_MEP[index] = te_4
#     recovery_20_all[index] = te_5
#     cd34_manual[index] = alpha
#     index = index+1
#
# plt.figure(figsize=(10,6))
# plt.plot(cd34_manual, recovery_20_HSC, '-o',color='r', label='HSC')
# plt.plot(cd34_manual, recovery_20_MPP, '-o',color='b', label='MPP')
# plt.plot(cd34_manual, recovery_20_CMP, '-o',color='g', label='CMP')
# plt.plot(cd34_manual, recovery_20_MEP, '-o',color='y', label='MEP')
# plt.plot(cd34_manual, recovery_20_all, '-o',color='k', label='All')
# #plt.ylim(0, 50)
# plt.legend()
# plt.show()


