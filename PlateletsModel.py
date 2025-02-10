import numpy as np
import math

class Thrombopoiesis:
    def __init__(self, parameters, k_pp=0, alpha_G=0, alpha_E=0):
        self.parameters = parameters
        self.k_pp = k_pp
        self.alpha_G = alpha_G
        self.alpha_E = alpha_E

    def MK_7(self, cells, t):
        # this is a healthy model for platelets which include seven different cell states and one residual platelets compartment.
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        k_decline = self.parameters["k_decline"]  # decline rate of platelets
        k_shed = self.parameters["k_shed"]  # shedding rate of megakaryocytes

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * (c[6]+c[7]))
        s_a = 1 / (1 + k_a * (c[6]+c[7]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # prim HSC
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]  # MPP
        dcdt[2] = (2 * a_c[2] * s_a - 1) * p_c[2] * s_p * c[2] + 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1]  # CMP
        dcdt[3] = (2 * a_c[3] * s_a - 1) * p_c[3] * s_p * c[3] + 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * c[2]  # MEP
        dcdt[4] = 2 * (1 - a_c[3] * s_a) * p_c[3] * s_p * c[3] - p_c[4] * c[4] # Mk-blast
        dcdt[5] = p_c[4] * c[4] - p_c[5] * c[5] # Mk
        dcdt[6] = p_c[5] * c[5] * k_shed - d_platelets * c[6]  # Platelet
        dcdt[7] = -k_decline* c[7]  # Residual Platelet
        return dcdt

    def component_7_extend(self, cells, t):
        # this is a healthy model for platelets include two branches for CFU-GEMM and MEP.
        # code haven't been completed or tested
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * c[6])
        s_a = 1 / (1 + k_a * c[6])

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # LT_HSC
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]  # ST_HSC
        dcdt[2] = (2 * a_c[2] * s_a - 1) * p_c[2] * s_p * c[2] + 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1]  # CMP
        dcdt[3] = (2 * a_c[3] * s_a - 1) * p_c[3] * s_p * c[3] + 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * c[2] * (1-self.alpha_G) # MEP
        dcdt[4] = 2 * (1 - a_c[3] * s_a) * p_c[3] * s_p * c[3] * (1-self.alpha_E) - p_c[4] * c[4]   # Mk-blast
        dcdt[5] = p_c[4] * c[4] - p_c[5] * c[5]     # Mk
        dcdt[6] = p_c[5] * c[5] * self.k_frac - d_platelets * c[6]  # Platelet
        return dcdt

    def component_7_Feedback(self, cells, t):
        # this is the model to see if the system has a constant feedback signal with the healthy steady state value
        # Both feedback signals are constant
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * c[6])
        s_a = 1 / (1 + k_a * c[6])

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # LT_HSC
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]  # ST_HSC
        dcdt[2] = (2 * a_c[2] * s_a - 1) * p_c[2] * s_p * c[2] + 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1]  # CMP
        dcdt[3] = (2 * a_c[3] * s_a - 1) * p_c[3] * s_p * c[3] + 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * c[2]  # MEP
        dcdt[4] = 2 * (1 - a_c[3] * s_a) * p_c[3] * s_p * c[3] - p_c[4] * c[4]  # Mk-blast
        dcdt[5] = p_c[4] * c[4] - p_c[5] * c[5]  # Mk
        dcdt[6] = p_c[5] * c[5] * self.k_frac - d_platelets * c[6]  # Platelet
        return dcdt

    def component_7_decline(self, cells, t):
        # this is designed to simulate the platelets decline from clinical data
        # add extra differential equation to simulate the decline.
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_decline = self.parameters["k_decline"] #decline ratio at the beginning of transplantation.
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * (c[6]+c[7]))
        s_a = 1 / (1 + k_a * (c[6]+c[7]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # LT_HSC
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]  # ST_HSC
        dcdt[2] = (2 * a_c[2] * s_a - 1) * p_c[2] * s_p * c[2] + 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1]  # CMP
        dcdt[3] = (2 * a_c[3] * s_a - 1) * p_c[3] * s_p * c[3] + 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * c[2]  # MEP
        dcdt[4] = 2 * (1 - a_c[3] * s_a) * p_c[3] * s_p * c[3] - p_c[4] * c[4]  # Mk-blast
        dcdt[5] = p_c[4] * c[4] - p_c[5] * c[5]  # Mk
        dcdt[6] = p_c[5] * c[5] * self.k_frac - d_platelets * c[6]  # Platelet
        dcdt[7] = -k_decline*c[7]
        return dcdt

    def component_6(self, cells, t):
        # six components: similar HPSC components with model from t.stiehl BMT(2014)
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))

        s_p = 1 / (1 + k_p * c[-1])
        s_a = 1 / (1 + k_a * c[-1])

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # prim. HSC
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a - 1) * s_p * p_c[1] * c[1]  # LTC-IC
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] + (2*a_c[2]*s_a - 1) * s_p * p_c[2] * c[2]  # MEP
        dcdt[3] = 2 * (1 - a_c[2] * s_a) * s_p * p_c[2] * c[2] - p_c[3] * c[3]  # Mk-blast
        dcdt[4] = p_c[3] * c[3] - p_c[4] * c[4]  # Mk
        dcdt[5] = self.k_frac * p_c[4] * c[4] - d_platelets * c[5]  # Platelets
        return dcdt

    def component_6_decline(self, cells, t):
        # six components model with platelets decline in clinc
        # add extra differential equation to simulate the decline.
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_decline = self.parameters["k_decline"]  # decline ratio at the beginning of transplantation.
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))

        s_p = 1 / (1 + k_p * (c[5]+c[6]))
        s_a = 1 / (1 + k_a * (c[5]+c[6]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a - 1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] + (2*a_c[2]*s_a - 1) * s_p * p_c[2] * c[2]
        dcdt[3] = 2 * (1 - a_c[2] * s_a) * s_p * p_c[2] * c[2] - p_c[3] * c[3]
        dcdt[4] = p_c[3] * c[3] - p_c[4] * c[4]
        dcdt[5] = self.k_frac * p_c[4] * c[4] - d_platelets * c[5]
        # Platelet
        dcdt[6] = -k_decline * c[6]
        return dcdt

    def component_6_feedback(self, cells, t):
        # six components model with platelets decline in clinc and also feedback affect also Mk-blast:
        # add extra differential equation to simulate the decline.
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_decline = self.parameters["k_decline"]  # decline ratio at the beginning of transplantation.
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))

        s_p = 1 / (1 + k_p * (c[5]+c[6]+c[4]))
        s_a = 1 / (1 + k_a * (c[5]+c[6]+c[4]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a - 1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] + (2*a_c[2]*s_a - 1) * s_p * p_c[2] * c[2]
        dcdt[3] = 2 * (1 - a_c[2] * s_a) * s_p * p_c[2] * c[2] - p_c[3] *s_p * c[3]
        dcdt[4] = p_c[3] * s_p * c[3] - p_c[4] * s_p * c[4]
        dcdt[5] = self.k_frac * p_c[4]* s_p * c[4] - d_platelets * c[5]
        # Platelet
        dcdt[6] = -k_decline * c[6]
        return dcdt

    def MK_5(self, cells, t):
        # five components: SC(Combine the apex SC till CMP), MEP, Mk-b, Mk, plt
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_decline = self.parameters["k_decline"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        k_frac = self.parameters['k_frac']

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        # s_p = 1 / (1 + k_p * (c[-1]+c[-2]))
        # s_a = 1 / (1 + k_a * (c[-1]+c[-2]))
        s_p = 1 / (1 + k_p * (c[2]+c[3]+c[4]+c[5]))
        s_a = 1 / (1 + k_a * (c[2]+c[3]+c[4]+c[5]))

        # dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        # dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2 * a_c[1] * s_a - 1) * s_p * p_c[1] * c[1]
        # dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2] * c[2]
        # dcdt[3] = p_c[2] * c[2] - p_c[3] * c[3]
        # dcdt[4] = self.k_frac * p_c[3] * c[3] - d_platelets * c[4]
        # dcdt[5] = -k_decline * c[5]

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a - 1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2] * c[2]
        dcdt[3] = p_c[2]* c[2] - p_c[3]* c[3]
        dcdt[4] = k_frac * p_c[3] * c[3] - d_platelets * c[4]
        dcdt[5] = -k_decline * c[5]
        return dcdt

    def MK_5_signal3(self, cells, t):
        # five components: SC(Combine the apex SC till CMP), MEP, Mk-b, Mk, plt
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_decline = self.parameters["k_decline"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        k_frac = self.parameters['k_frac']

        c = np.array(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * (c[2]+c[3]+c[4]+c[5]))
        s_a = 1 / (1 + k_a * (c[2]+c[3]+c[4]+c[5]))

        # dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        # dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2 * a_c[1] * s_a - 1) * s_p * p_c[1] * c[1]
        # dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2] * c[2]
        # dcdt[3] = p_c[2] * c[2] - p_c[3] * c[3]
        # dcdt[4] = self.k_frac * p_c[3] * c[3] - d_platelets * c[4]
        # dcdt[5] = -k_decline * c[5]

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a - 1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2] * s_p * c[2]
        dcdt[3] = p_c[2] * s_p * c[2] - p_c[3] * s_p * c[3]
        dcdt[4] = k_frac * p_c[3] * s_p * c[3] - d_platelets * c[4]
        dcdt[5] = -k_decline * c[5]
        return dcdt

    def component_4_equilibrium(self, cells, t):
        # simplify the model into 4 output by substituting with the equilibrium state from compartment seven
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * c[-1])
        s_a = 1 / (1 + k_a * c[-1])

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]
        dcdt[2] = (2 * a_c[3] * s_a - 1) * p_c[3] * s_p * c[2] + 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * 2 * (1 - a_c[1] * s_a) * p_c[1] * c[1] / ((1 - 2*a_c[2]*s_a)*p_c[2])
        dcdt[3] = self.k_frac * 2 * (1 - a_c[3]*s_a) * p_c[3]*s_p*c[2] - d_platelets * c[3]
        return dcdt

    def MK_4(self, cells, t):
        # four component: stem and Pro, Immature Mk, mature MK, Plt.
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        k_decline = self.parameters["k_decline"]  # decline ratio at the beginning after transplantation

        s_p = 1 / (1 + k_p * (c[-1]+c[-2]))
        s_a = 1 / (1 + k_a * (c[-1]+c[-2]))

        dcdt[0] = (2 * a_c * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c * s_a) * p_c[0] * s_p * c[0] - p_c[1]*c[1]
        dcdt[2] = p_c[1]*c[1] - p_c[2]*c[2]
        dcdt[3] = self.k_frac * p_c[2]*c[2] - d_platelets*c[3]
        dcdt[4] = -k_decline * c[4]

        return dcdt

    def simplify_2(self, cells, t):
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        # two components model to test some algorithm
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets

        s_p = 1 / (1 + k_p * c[-1])
        s_a = 1 / (1 + k_a * c[-1])

        dcdt[0] = (2 * a_c * s_a - 1) * p_c * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c * s_a) * p_c * s_p * c[0] * self.k_frac- d_platelets * c[1]
        return dcdt

    def simplify_2_decline(self, cells, t):
        # two components model to test some algorithm
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_decline = self.parameters["k_decline"]  # decline ratio at the beginning of transplantation.
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * (c[-1]+c[-2]))
        s_a = 1 / (1 + k_a * (c[-1]+c[-2]))

        dcdt[0] = (2 * a_c * s_a - 1) * p_c * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c * s_a) * p_c * s_p * c[0] * self.k_frac- d_platelets * c[1]
        dcdt[2] = -k_decline * c[2]
        return dcdt

    def simplify_3(self, cells, t):
        # three components model to test some algorithm
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        d_platelets = self.parameters["d_plt"]  # clearance of mature platelets
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))
        s_p = 1 / (1 + k_p * c[-1])
        s_a = 1 / (1 + k_a * c[-1])

        dcdt[0] = (2 * a_c * s_a - 1) * p_c[0]* s_p * c[0]
        dcdt[1] = 2 * (1 - a_c * s_a) * p_c[0]* s_p * c[0] - p_c[1] * c[1]
        dcdt[2] = self.k_frac * p_c[1] * c[1] - d_platelets * c[2]
        return dcdt


class Mutation:
    def __init__(self, parameters, alpha_G=0, alpha_E=0):
        self.parameters = parameters
        self.alpha_G = alpha_G
        self.alpha_E = alpha_E

    def MPN_2(self, cells, t):
        # Simpliest model to simulate MPN:
        c = np.asarray(cells)
        dcdt = np.zeros(len(c))

        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        d_plt = self.parameters["d_plt"]  # clearance of mature platelets
        p_l = self.parameters["p_l"]
        a_l = self.parameters["a_l"]
        d_mpn = self.parameters["d_mpn"]

        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_pm = self.parameters["k_pm"]
        k_am = self.parameters["k_am"]

        s_p = 1 / (1 + k_p * (c[1]+c[3]))
        s_a = 1 / (1 + k_a * (c[1]+c[3]))
        s_pm = 1 / (1 + k_pm * (c[1]+c[3]))
        s_am = 1 / (1 + k_am * (c[1]+c[3]))

        dcdt[0] = (2 * a_c * s_a - 1) * p_c * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c * s_a) * p_c * s_p * c[0] * self.k_frac - d_plt * c[1]
        dcdt[2] = (2 * a_l * s_am - 1) * p_l * s_pm * c[2] - d_mpn * c[2]
        dcdt[3] = 2 * (1 - a_l *s_am) * p_l *s_pm * c[2] * self.k_frac - d_mpn * c[3]
        return dcdt

    def MPN_4(self, cells, t): #__(not finished)__
        # four components: SC, Progenitor, MK, plt
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        d_plt = self.parameters["d_plt"]  # clearance of mature platelets
        p_l = self.parameters["p_l"]
        a_l = self.parameters["a_l"]
        d_mpn = self.parameters["d_mpn"]

        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_p_mpn = self.parameters["k_p_mpn"]
        k_a_mpn = self.parameters["k_a_mpn"]

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))

        s_p = 1 / (1 + k_p * (c[3]+c[7]))
        s_a = 1 / (1 + k_a * (c[3]+c[7]))

        s_p_mpn = 1 / (1 + k_p_mpn * (c[3]+c[7]))
        s_a_mpn = 1 / (1 + k_a_mpn * (c[3]+c[7]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a-1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2] * c[2]
        dcdt[3] = self.k_frac * p_c[2] * c[2] - d_plt * c[3]

        dcdt[4] = (2 * a_l[0] * s_a_mpn - 1) * p_l[0] * s_p_mpn * c[4]
        dcdt[5] = 2 * (1 - a_l[0] * s_a_mpn) * p_l[0] * s_p_mpn * c[4] + (2 * a_l[1]* s_a_mpn - 1) * s_p_mpn * p_l[1] * c[5]
        dcdt[6] = 2 * (1 - a_l[1] * s_a_mpn) * p_l[1] * s_p_mpn * c[5] - p_l[2] * c[6]
        dcdt[7] = self.k_frac * p_l[2] * c[6] - d_mpn * c[7]
        return dcdt

    def MPN_5(self, cells, t):
        # four components: SC, Progenitor, MK, plt
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        d_plt = self.parameters["d_plt"]  # clearance of mature platelets
        p_l = self.parameters["p_l"]
        a_l = self.parameters["a_l"]
        k_frac = self.parameters["k_frac"]
        d_mpn = self.parameters["d_mpn"]

        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_p_mpn = self.parameters["k_p_mpn"]
        k_a_mpn = self.parameters["k_a_mpn"]
        k_frac_mpn = self.parameters['k_frac_mpn']

        c = np.asarray(cells[:5])
        l = np.asarray(cells[5:])
        dcdt = np.zeros(len(cells))
        # k_frac_mpn = 5000

        s_p = 1 / (1 + k_p * (c[-1]+l[-1]))
        s_a = 1 / (1 + k_a * (c[-1]+l[-1]))

        s_p_mpn = 1 / (1 + k_p_mpn * (c[-1]+l[-1]))
        s_a_mpn = 1 / (1 + k_a_mpn * (c[-1]+l[-1]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a-1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2] * s_p * c[2]
        dcdt[3] = p_c[2] * s_p * c[2] - p_c[3] * s_p * c[3]
        dcdt[4] = k_frac * p_c[3] * s_p * c[3] - d_plt * c[4]

        dcdt[5] = (2 * a_l[0] * s_a_mpn - 1) * p_l[0] * s_p_mpn * l[0]
        dcdt[6] = 2 * (1 - a_l[0] * s_a_mpn) * p_l[0] * s_p_mpn * l[0] + (2 * a_l[1]* s_a_mpn - 1) * s_p_mpn * p_l[1] * l[1]
        dcdt[7] = 2 * (1 - a_l[1] * s_a_mpn) * p_l[1] * s_p_mpn * l[1] - p_l[2] * s_p_mpn * l[2]
        dcdt[8] = p_l[2] * s_p_mpn * l[2] - p_l[3] * s_p_mpn * l[3]
        dcdt[9] = k_frac_mpn * p_l[3] * s_p_mpn * l[3] - d_mpn * l[4]
        return dcdt

    def MPN_5_signal3(self, cells, t):
        # five components: HSC, HPSC, MKb, MK, Plt.
        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        d_plt = self.parameters["d_plt"]  # clearance of mature platelets
        p_l = self.parameters["p_l"]
        a_l = self.parameters["a_l"]
        k_frac = self.parameters["k_frac"]
        d_mpn = self.parameters["d_mpn"]

        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_p_mpn = self.parameters["k_p_mpn"]
        k_a_mpn = self.parameters["k_a_mpn"]
        k_frac_mpn = self.parameters['k_frac_mpn']

        c = np.asarray(cells[:5])
        l = np.asarray(cells[5:])
        dcdt = np.zeros(len(cells))
        # k_frac_mpn = 5000

        s_p = 1 / (1 + k_p * (c[2]+c[3]+c[4]+l[2]+l[3]+l[4]))
        s_a = 1 / (1 + k_a * (c[2]+c[3]+c[4]+l[2]+l[3]+l[4]))

        s_p_mpn = 1 / (1 + k_p_mpn * (c[-1]+l[-1]))
        s_a_mpn = 1 / (1 + k_a_mpn * (c[-1]+l[-1]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]
        dcdt[1] = 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0] + (2*a_c[1]*s_a-1) * s_p * p_c[1] * c[1]
        dcdt[2] = 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1] - p_c[2]  * c[2]
        dcdt[3] = p_c[2] * c[2] - p_c[3] * c[3]
        dcdt[4] = k_frac * p_c[3] * c[3] - d_plt * c[4]

        dcdt[5] = (2 * a_l[0] * s_a_mpn - 1) * p_l[0] * s_p_mpn * l[0]
        dcdt[6] = 2 * (1 - a_l[0] * s_a_mpn) * p_l[0] * s_p_mpn * l[0] + (2 * a_l[1] * s_a_mpn - 1) * s_p_mpn * p_l[1] * l[1]
        dcdt[7] = 2 * (1 - a_l[1] * s_a_mpn) * p_l[1] * s_p_mpn * l[1] - p_l[2] * l[2]
        dcdt[8] = p_l[2] * l[2] - p_l[3] * l[3]
        dcdt[9] = k_frac_mpn * p_l[3] * l[3] - d_mpn * l[4]
        return dcdt

    def CALR_5(self, cells, t):
        p_l = self.parameters[0,:]
        a_l = self.parameters[1,:]
        d_mpn = self.parameters[2,-1]
        # k_p_mpn = self.parameters["k_p_mpn"]
        # k_a_mpn = self.parameters["k_a_mpn"]
        l = np.asarray(cells)
        dcdt = np.zeros(len(cells))
        # s_p_mpn = 1 / (1 + k_p_mpn * (c[-1]+l[-1]))
        # s_a_mpn = 1 / (1 + k_a_mpn * (c[-1]+l[-1]))

        dcdt[0] = (2 * a_l[0] - 1) * p_l[0] * l[0]
        dcdt[1] = 2 * (1 - a_l[0]) * p_l[0]* l[0] + (2 * a_l[1] - 1)  * p_l[1] * l[1]
        dcdt[2] = 2 * (1 - a_l[1]) * p_l[1] * l[1] - p_l[2] * l[2]
        dcdt[3] = p_l[2] * l[2] - p_l[3] * l[3]
        dcdt[4] = self.k_frac * p_l[3] * l[3] - d_mpn * l[4]
        return dcdt

    def MPN_6(self, cells, t):
        # this is a model without feedback signal, try to model when CALR mutate,
        # JAK-STAT signalling keep activating and platelet formation is independent from TPO.
        c = np.asarray(cells[0:6])
        l = np.asarray(cells[6:12])
        dcdt = np.zeros(len(c) + len(l))

        p_c = self.parameters["p_c"]  # proliferation rate
        a_c = self.parameters["a_c"]  # self-renewal
        d_plt = self.parameters["d_plt"]  # clearance of mature platelets
        p_l = self.parameters["p_l"]
        a_l = self.parameters["a_l"]
        d_mpn = self.parameters["d_mpn"]

        k_p = self.parameters["k_p"]
        k_a = self.parameters["k_a"]
        k_p_mpn = self.parameters["k_p_mpn"]
        k_a_mpn = self.parameters["k_a_mpn"]

        s_p = 1 / (1 + k_p * (c[5] + l[5]))
        s_a = 1 / (1 + k_a * (c[5] + l[5]))
        s_p_mpn = 1 / (1 + k_p_mpn * (c[5] + l[5]))
        s_a_mpn = 1 / (1 + k_a_mpn * (c[5] + l[5]))

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # LT_HSC
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]  # ST_HSC
        dcdt[2] = (2 * a_c[2] * s_a - 1) * p_c[2] * s_p * c[2] + 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1]  # CMP
        dcdt[3] = 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * c[2] - p_c[3] * c[3]  # Mk-blast
        dcdt[4] = p_c[3] * c[3] - p_c[4] * c[4]  # Mk
        dcdt[5] = p_c[4] * c[4] * self.k_frac - d_plt * c[5]  # Platelet

        dcdt[6] = (2 * a_l[0] * s_a_mpn- 1) * p_l[0] * s_p_mpn * l[0]  # mLT_HSC
        dcdt[7] = (2 * a_l[1] * s_a_mpn- 1) * p_l[1] * s_p_mpn * l[1] + 2 * (1 - a_l[0] * s_a_mpn) * p_l[0] * s_p_mpn * l[0]  # mST_HSC
        dcdt[8] = (2 * a_l[2] * s_a_mpn - 1) * p_l[2] * s_p_mpn * l[2] + 2 * (1 - a_l[1] * s_a_mpn) * p_l[1] * s_p_mpn * l[1]  # mCMP
        dcdt[9] = 2 * (1 - a_l[2] * s_a_mpn) * p_l[2] * s_p_mpn * l[2] - p_l[3] * l[3]  # mMk-blast
        dcdt[10] = p_l[3] * l[3] - p_l[4] * l[4]  # mMk
        dcdt[11] = p_l[4] * l[4] * self.k_frac - d_mpn * l[5]  # mPlatelet
        return dcdt


class Leukocyte:
    def __init__(self, parameters, k_pw, k_aw, d_w):
        self.parameters = parameters
        self.k_pw = k_pw
        self.k_aw = k_aw
        self.d_w = d_w
    def WBC_Formation(self, cells, t):
        # this is white blood cells model from stiehl, et al.
        p_w = self.parameters[0, :]  # proliferation rate
        a_w = self.parameters[1, :]  # self-renewal

        c = np.asarray(cells)
        dcdt = np.zeros(len(c))

        s_p = 1 / (1 + self.k_pw * c[6])
        s_a = 1 / (1 + self.k_aw * c[6])

        dcdt[0] = (2 * a_w[0] * s_a - 1) * p_w[0] * s_p * c[0]  # prim. HSC
        dcdt[1] = (2 * a_w[1] * s_a - 1) * p_w[1] * s_p * c[1] + 2 * (1 - a_w[0] * s_a) * p_w[0] * s_p * c[0]  # LTC-IC
        dcdt[2] = (2 * a_w[2] * s_a - 1) * p_w[2] * s_p * c[2] + 2 * (1 - a_w[1] * s_a) * p_w[1] * s_p * c[1]  # CFU-GM
        dcdt[3] = (2 * a_w[3] * s_a - 1) * p_w[3] * s_p * c[3] + 2 * (1 - a_w[2] * s_a) * p_w[2] * s_p * c[2]  # CFU-G
        dcdt[4] = (2 * a_w[4] * s_a - 1) * p_w[4] * c[4] + 2 * (1 - a_w[3] * s_a) * p_w[3] * s_p * c[3]  # Myeloblast
        dcdt[5] = (2 * a_w[5] * s_a - 1) * p_w[5] * c[5] + 2 * (1 - a_w[4] * s_a) * p_w[4] * c[4]  # Promyelocyte
        dcdt[6] = (2 * a_w[6] * s_a - 1) * p_w[6] * c[6] + 2 * (1 - a_w[5] * s_a) * p_w[5] * c[5]  # Myelocyte
        dcdt[7] = 2 * (1 - a_w[6] * s_a) * p_w[6] * c[6] - (self.d_w /s_p) * c[7]  # circulating neutrophils  circulating leukocytes
        return dcdt

class iPSC:
    def __init__(self, paras):
        self.paras = paras

    def ODE_logistic(self, cells, t):
        x = np.asarray(cells)
        r = self.paras[0]
        K = self.paras[1]
        s_p = (1 - x / K)
        dydt = r * x * s_p
        return dydt

    def ODE_Gompertz(self, cells, t):
        x = np.asarray(cells)
        r = self.paras[0]
        K = self.paras[1]
        dydt = r*x*np.log(K/x)
        return dydt

    def ODE_aml(self, cells, t):
        # use the signal like niche-related feedback
        x = np.asarray(cells)
        r = self.paras[0]
        K = self.paras[1]
        s_p = ((K - x) / (1 + K - x)) / (K / (1 + K))
        dydt = r * x * s_p
        return dydt

    def ODE3_logistic(self, cells, t):
        N1, N2, N3 = cells

        K = self.paras[0]
        r1, r2, r3 = self.paras[1:4]
        a1, a2 = self.paras[4:6]
        d1, d2, d3 = self.paras[6:]
        N_total = N1 + N2 + N3
        H = 1 if N_total > K else 0  # Heaviside step function
        dN1_dt = r1 * N1 * (1 - N_total / K) - a1 * N1 - d1 * H * N1
        dN2_dt = r2 * N2 * (1 - N_total / K) + a1 * N1 - a2 * N2 - d2 * H * N2
        dN3_dt = r3 * N3 * (1 - N_total / K) + a2 * N2 - d3 * H * N3
        return [dN1_dt, dN2_dt, dN3_dt]

    def ODE4_logistic(self, cells, t):
        N1, N2, N3, N4 = cells
        K = self.paras[0]
        r1, r2, r3, r4 = self.paras[1:5]
        a1, a2, a3 = self.paras[5:]
        N_total = N1+N2+N3+N4
        dN1_dt = r1 * N1 * (1 - N_total / K) - a1 * N1
        dN2_dt = r2 * N2 * (1 - N_total / K) + a1 * N1 - a2 * N2
        dN3_dt = r3 * N3 * (1 - N_total / K) + a2 * N2 - a3 * N3
        dN4_dt = r4 * N4 * (1 - N_total / K) + a3 * N3
        return [dN1_dt, dN2_dt, dN3_dt, dN4_dt]

    def ODE4_MPN(self, cells, t):
        c0, c1, c2, c3 = cells
        K = 20000
        p0, p1, p2 = self.paras[0:3]
        a0 = self.paras[3]
        a1 = self.paras[4]
        d3 = self.paras[5]
        N_total = c0 + c1 + c2 + c3
        # H = 1 if N_total > K else 0  # Heaviside step function
        s = 0
        # s = 1 - N_total / K
        # s apply to a
        # dc0_dt = (2*a0*s-1)*p0*c0
        # dc1_dt = 2*(1-a0*s)*p0*c0 + (2*a1*s-1)*p1*c1
        # dc2_dt = 2*(1-a1*s)*p1*c1 - p2*c2
        # dc3_dt = p2*c2 - d3*c3
        # # s apply to p
        # dc0_dt = (2*a0-1)*p0*s*c0
        # dc1_dt = 2*(1-a0)*p0*s*c0 + (2*a1*s-1)*p1*s*c1
        # dc2_dt = 2*(1-a1)*p1*s*c1 - p2*s*c2
        # dc3_dt = p2*s*c2  -d3*c3
        dc0_dt = (2 * a0 * s - 1) * p0 * c0
        dc1_dt = 2 * (1 - a0 * s) * p0 * c0 + (2 * a1 * s - 1) * p1 * c1
        dc2_dt = 2 * (1 - a1 * s) * p1 * c1 - p2 * c2
        dc3_dt = p2 * c2 - d3 * c3
        return [dc0_dt, dc1_dt, dc2_dt, dc3_dt]

    def ODE4_MPNm(self, cells, t):
        # same ODE model but with memory of the cells:
        c0, c1, c2, c3, c0_tilde, c1_tilde, c2_tilde, c3_tilde = cells
        K = 20000
        p0, p1, p2 = self.paras[0:3]
        a0, a1 = self.paras[3:5]
        d3 = self.paras[5]
        N_total = c0 + c1 + c2 + c3
        N_total_d = c0_tilde + c1_tilde + c2_tilde + c3_tilde
        H = 1 if N_total > K else 0
        s = 0 if (1 - N_total/K < 0) else (1 - N_total/K)

        #s = 1 - N_total/K
        #s = np.log(K/N_total) # Gompertz Growth
        # s applied on a:
        dc0_dt = (2 * a0 * s - 1) * p0 * c0
        dc1_dt = 2 * (1 - a0 * s) * p0 * c0 + (2 * a1 * s - 1) * p1 * c1
        dc2_dt = 2 * (1 - a1 * s) * p1 * c1 - p2 * c2
        dc3_dt = p2 * c2
        dc0_tilde_dt = (2 * a0 * s - 1) * p0 * c0_tilde
        dc1_tilde_dt = 2 * (1 - a0 * s) * p0 * c0_tilde + (2 * a1 * s - 1) * p1 * c1_tilde
        dc2_tilde_dt = 2 * (1 - a1 * s) * p1 * c1_tilde - p2 * c2_tilde
        dc3_tilde_dt = p2 * c2_tilde - d3*c3_tilde*H
        # s applied on p:
        # dc0_dt = (2 * a0 - 1) * p0 *s_p* c0
        # dc1_dt = 2 * (1 - a0) * p0 *s_p* c0 + (2 * a1 - 1) * p1*s_p* c1
        # dc2_dt = 2 * (1 - a1) * p1 *s_p* c1 - p2 *s_p* c2
        # dc3_dt = p2 *s_p* c2
        # dc0_tilde_dt = (2 * a0 - 1) * p0 *s_p* c0_tilde
        # dc1_tilde_dt = 2 * (1 - a0) * p0 *s_p* c0_tilde + (2 * a1 - 1) * p1 *s_p* c1_tilde
        # dc2_tilde_dt = 2 * (1 - a1) * p1 *s_p* c1_tilde - p2 *s_p* c2_tilde
        # dc3_tilde_dt = p2 *s_p* c2_tilde - d3*c3_tilde
        return [dc0_dt, dc1_dt, dc2_dt, dc3_dt, dc0_tilde_dt, dc1_tilde_dt, dc2_tilde_dt, dc3_tilde_dt]
