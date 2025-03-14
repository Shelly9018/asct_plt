import numpy as np

class Thrombopoiesis:
    def __init__(self, parameters):
        self.parameters = parameters

    def Plt_7(self, cells, t):
        # this is an ode model for platelets engraftment, which includes seven different cell states and one residual platelets compartment.
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

        dcdt[0] = (2 * a_c[0] * s_a - 1) * p_c[0] * s_p * c[0]  # HSC
        dcdt[1] = (2 * a_c[1] * s_a - 1) * p_c[1] * s_p * c[1] + 2 * (1 - a_c[0] * s_a) * p_c[0] * s_p * c[0]  # MPP
        dcdt[2] = (2 * a_c[2] * s_a - 1) * p_c[2] * s_p * c[2] + 2 * (1 - a_c[1] * s_a) * p_c[1] * s_p * c[1]  # CMP
        dcdt[3] = (2 * a_c[3] * s_a - 1) * p_c[3] * s_p * c[3] + 2 * (1 - a_c[2] * s_a) * p_c[2] * s_p * c[2]  # MEP
        dcdt[4] = 2 * (1 - a_c[3] * s_a) * p_c[3] * s_p * c[3] - p_c[4] * c[4] # Mk-blast
        dcdt[5] = p_c[4] * c[4] - p_c[5] * c[5] # Mk
        dcdt[6] = p_c[5] * c[5] * k_shed - d_platelets * c[6]  # Platelet
        dcdt[7] = -k_decline* c[7]  # Residual Platelet
        return dcdt
