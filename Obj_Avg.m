function obj = Obj_Avg(p)
load('Average_plt.mat')
load('Days.mat')
p_c = p(1,1:6); % p_c(5) is e5, p_c(6) is e6
a_c = [p(1,7:10), 0, 0];
k_p = p(1,11)*10^(-10);
d_plt = p(1,12);
k_shed = p(1,13)*1000;
k_decline = 0.345;
c_plt = 1.4e10;
CD34 = 3.5e6;
plt_start = Average_plt(1)*5e9/70;
k_a = (2*a_c(1)-1) / c_plt;
parameters = [p_c;a_c;k_p,k_a,d_plt,k_decline,k_shed, 0];

% Healthy equilibrium:
c_equ(6) = (d_plt*c_plt) / (k_shed * p_c(6));
c_equ(5) = (d_plt*c_plt) / (k_shed * p_c(5));
c_equ(4) = (d_plt*c_plt * (1 + k_p * c_plt) * a_c(1))/ (k_shed * (2 * a_c(1) - a_c(4)) * p_c(4));
c_equ(3) = (d_plt*c_plt * (1 + k_p * c_plt) * (a_c(1) - a_c(4)) * a_c(1)) / (k_shed * (2 * a_c(1) - a_c(4)) * (2 * a_c(1) - a_c(3)) * p_c(3));
c_equ(2) = (d_plt*c_plt * (1 + k_p * c_plt) * (a_c(1) - a_c(3)) * (a_c(1) - a_c(4)) * a_c(1)) / (k_shed * (2 * a_c(1) - a_c(4)) * (2 * a_c(1) - a_c(3)) * (2 * a_c(1) - a_c(2)) * p_c(2));
c_equ(1) = (d_plt*c_plt * (1 + k_p * c_plt) * (a_c(1) - a_c(2)) * (a_c(1) - a_c(3)) * (a_c(1) - a_c(4))) / (k_shed * (2 * a_c(1) - a_c(4)) * (2 * a_c(1) - a_c(3)) * (2 * a_c(1) - a_c(2)) * p_c(1));
obj_equ = ((c_equ(6)-1.6e7)/1e7)^2+((c_equ(5)-1.6e7)/1e7)^2+((c_equ(4)-2e7)/1e7+(c_equ(3)-2e7)/1e7)^2+((c_equ(2)-4e6)/1e6)^2+((c_equ(1)-1.4e5)/1e5)^2;

tspan = [0 200]; % days
c0 = [CD34*0.0408, CD34*0.072, CD34*0.284, CD34*0.148, 0, 0, 0, plt_start];
[t, c] = ode45(@(t,c) ODE7( t, c, parameters), tspan, c0);
obj_sim = 0;
for i = 1:length(Days)
   [row(i), col(i)] = find(t>=Days(i), 1);
   obj_temp = c(row(i), 7)+c(row(i), 8) - Average_plt(i)*5e9/70;
   obj_sim = obj_sim+(obj_temp/1e10)^2;
end

obj = obj_equ/6 + obj_sim/length(Days);
end