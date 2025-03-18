close all;
clear;

% set your own path to save the optimization results:
path = 'Your Path';

% load average patient data:
Patient = readmatrix('../data/data_average.csv');
Days = Patient(30:end,1);
Average_plt = Patient(30:end,2);
CD34 = 3.5e6;
k_decline = 0.345;
plt_start = Average_plt(1)*5e9/70;

% Bound Constraints:
A = [0,0,0,0,0,0,-1,1,0,0,0,0,0; % a1>a2
    0,0,0,0,0,0,-1,0,1,0,0,0,0;  % a1>a3
    0,0,0,0,0,0,-1,0,0,1,0,0,0;  % a1>a4
    1,-1,0,0,0,0,0,0,0,0,0,0,0;  % p1<p2
    0,1,-1,0,0,0,0,0,0,0,0,0,0;  % p2<p3
    0,0,1,-1,0,0,0,0,0,0,0,0,0]; % p3<p4
b = [0;0;0;0;0;0];
Aeq = [];
beq = [];
lb = [0.001 0.001 0.0  0.0  0.0 0.0 0.5 0.0 0.0 0.0, 2.0, 0.0, 1.0]; % lower bound
ub = [0.05  0.2   2.0  2.0  2.0 2.0 1.0 1.0 1.0 1.0, 15,  0.2, 5.0];  % upper bound

% Use fmincon to do 1000 times parameters optimization:
for i = 1:2
    % Generate random 1000 Parameter Initial Guess:
    p = [lb(1)+rand(1)*(ub(1)-lb(1)), lb(2)+rand(1)*(ub(2)-lb(2)), lb(3)+rand(1)*(ub(3)-lb(3)), lb(4)+rand(1)*(ub(4)-lb(4)), lb(5)+rand(1)*(ub(5)-lb(5)), lb(6)+rand(1)*(ub(6)-lb(6))];
    a = [lb(7)+rand(1)*(ub(7)-lb(7)), lb(8)+rand(1)*(ub(8)-lb(8)), lb(9)+rand(1)*(ub(9)-lb(9)), lb(10)+rand(1)*(ub(10)-lb(10))];
    k_p = lb(11)+rand(1)*(ub(11)-lb(11));
    d_plt = lb(12)+rand(1)*(ub(12)-lb(12));
    k_shed = lb(13)+rand(1)*(ub(13)-lb(13));
    % Initial guess parameter matrix:
    parameters0 = [p,a,k_p,d_plt,k_shed];
    % Healthy equilibrium state platelets count in average:
    c_plt = 1.4*10^10;

    % minimize the objective function with fmincon and save initial
    % parameter, results parameter, minimal objective function value in the
    % target address.
    [parameters, fval] = fmincon(@(p) Obj_Avg(p),parameters0,A,b,Aeq,beq,lb,ub);
    save([path,num2str(i),'.mat'],'parameters0','parameters','fval');

    % reformat the parameters to solve ODE:
    para_set = zeros([3, 6]);
    para_set(1,:) = parameters(1,1:6);
    para_set(2,1:4) = parameters(1,7:10);
    para_set(3,1:5) = [parameters(1, 11)*10^(-10), (2*parameters(1, 7)-1)/c_plt, parameters(1, 12), k_decline, parameters(1,13)*1000];
    tspan = [0 1000];
    c0 = [CD34*0.0408, CD34*0.072, CD34*0.284, CD34*0.148, 0, 0, 0, plt_start];
    [t,c] = ode45(@(t, c)  ODE( t, c, para_set), tspan, c0);
    
    % save simulation result figures:
    f = figure('visible','off');
    scatter(Days, Average_plt);
    hold on
    xlabel('Time [days]','FontWeight','bold');
    ylabel('Platelets [/nl]','FontWeight','bold');
    xlim([-30 150]);
    plot(t, (c(:,7)+c(:,8))*70/5e9,'r','LineWidth',2);
    legend('Clinical data','Simulation result')
    hold off
    exportgraphics(f, [path, num2str(i), '.png'])
end