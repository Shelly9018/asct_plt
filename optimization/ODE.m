function dcdt = ODE(t, c, p)

% extract parameters from input matrix p:
p_c = p(1,:); % p_c(1:4) is p1,p2,p3,p4, proliferation rate of HSC, MPP, CMP, and MEP; p_c(5:6) is e5, e6 the duration of endomitosis of MKb and MK.
a_c = p(2,:); % a_c(1:4) is a1,a2,a3,a4, self-renewal fraction of HSC, MPP, CMP, and MEP; a_c(5:6) is empty and set to be zero.
k_p = p(3,1); % kp, feedback signal parameter.
k_a = p(3,2); % ka, feedback signal parameter.
d_plt = p(3,3); % clearance rate of platelets.
k_decline = p(3,4); % clearance rate of the residual platelets before diagnosis. 
k_shed = p(3,5); % one megakaryocyte can shed into k_shed number of platelets.

% feedback signal: (Equ.9 to 10)
s_p = 1/(1+k_p*(c(7,1)+c(8,1))); 
s_a = 1/(1+k_a*(c(7,1)+c(8,1)));

% initialize ODE system output:
dcdt = zeros(7,1);

% ODE equation: (Equ.1 to 8)
dcdt(1,1) = (2*a_c(1)*s_a - 1)*p_c(1)*s_p*c(1,1);
dcdt(2,1) = (2*a_c(2)*s_a  - 1)*p_c(2)*s_p*c(2,1) + 2*(1-a_c(1)*s_a)*p_c(1)*s_p*c(1,1);
dcdt(3,1) = (2*a_c(3)*s_a - 1)*p_c(3)*s_p*c(3,1) + 2*(1-a_c(2)*s_a)*p_c(2)*s_p*c(2,1);
dcdt(4,1) = (2*a_c(4)*s_a - 1)*p_c(4)*s_p*c(4,1) + 2*(1-a_c(3)*s_a)*p_c(3)*s_p*c(3,1);
dcdt(5,1) = 2*(1-a_c(4)*s_a)*p_c(4)*s_p*c(4,1) - p_c(5)*c(5,1);
dcdt(6,1) = p_c(5)*c(5,1)-p_c(6)*c(6,1);
dcdt(7,1) = k_shed*p_c(6)*c(6,1) - d_plt*c(7,1);
dcdt(8,1) = -k_decline*c(8,1);
end