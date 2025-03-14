close all;
clear;

path = 'E:\BMT_Optimization\Results_Pat44\';
% load patient data:
load('CD34.mat');
load('Patients.mat');
id = 44;
index = ~isnan(Patients(:,id+1));
Patients_plt = Patients(index,id+1);
Days = Patients(index,1);
filter = find(Days>=0, 1);
Patients_plt0 = Patients_plt(filter:end);
Days0 = Days(filter:end);
CD34_patient = CD34(id)*1e6;

% fit to get k_decline:
f = fit(Days0(1:4),Patients_plt0(1:4),'exp1');
%plt_start = f.a*5e9/70;
plt_start = Patients_plt0(1)*5e9/70;
k_decline = -f.b;
% Bound Constraints:
A = [0,0,0,0,0,0,-1,1,0,0,0,0,0;
    0,0,0,0,0,0,-1,0,1,0,0,0,0;
    1,-1,0,0,0,0,0,0,0,0,0,0,0;
    0,1,-1,0,0,0,0,0,0,0,0,0,0;
    0,0,1,-1,0,0,0,0,0,0,0,0,0;
    0,0,0,0,0,0,-1,0,0,1,0,0,0]; % 0,0,0,0,0,0,0,0,-1,1,0,0,0;
b = [0;0;0;0;0;0];
Aeq = [];
beq = [];
lb = [0.001 0.001 0.0  0.0  0.0 0.1 0.5 0.0 0.0 0.0, 2.0, 0.0, 1.0];
ub = [0.05  0.2   2.0  2.0  2.0 2.0 1.0 1.0 1.0 1.0, 15, 0.2, 5.0];

% Generate random 100 Parameter Initial Guess:
for i = 1:500
    p = [lb(1)+rand(1)*(ub(1)-lb(1)), lb(2)+rand(1)*(ub(2)-lb(2)), lb(3)+rand(1)*(ub(3)-lb(3)), lb(4)+rand(1)*(ub(4)-lb(4)), lb(5)+rand(1)*(ub(5)-lb(5)), lb(6)+rand(1)*(ub(6)-lb(6))];
    a = [lb(7)+rand(1)*(ub(7)-lb(7)), lb(8)+rand(1)*(ub(8)-lb(8)), lb(9)+rand(1)*(ub(9)-lb(9)), lb(10)+rand(1)*(ub(10)-lb(10))];
    k_p = lb(11)+rand(1)*(ub(11)-lb(11));
    d_plt = lb(12)+rand(1)*(ub(12)-lb(12));
    k_shed = lb(13)+rand(1)*(ub(13)-lb(13));
    parameters0 = [p,a,k_p,d_plt,k_shed];
    
    % Other random generate parameters:
    c_plt = (1+rand(1)*(3.2-1))*10^10;
    [parameters, fval] = fmincon(@(p) Obj_patients(p, k_decline, id, c_plt, plt_start),parameters0,A,b,Aeq,beq,lb,ub);
    save([path,num2str(i),'.mat'],'parameters');
    % plot the fitting results:
    para_set = zeros([3, 6]);
    para_set(1,:) = parameters(1,1:6);
    para_set(2,1:4) = parameters(1,7:10);
    para_set(3,1:5) = [parameters(1, 11)*10^(-10), (2*parameters(1, 7)-1)/c_plt, parameters(1, 12), k_decline, parameters(1,13)*1000];
    tspan = [0 1000];
    c0 = [CD34_patient*0.0408, CD34_patient*0.072, CD34_patient*0.284, CD34_patient*0.148, 0, 0, 0, plt_start];
    [t,c] = ode45(@(t, c)  ODE7( t, c, para_set), tspan, c0);
    f = figure('visible','off');
    plot(Days0, Patients_plt0*5e9/70, 'b-o');
    hold on
    xlabel('Time [days]','FontWeight','bold');
    ylabel('Platelets [/kg]','FontWeight','bold');
    xlim([-30 150]);
    plot(t, c(:,7)+c(:,8),'r','LineWidth',2);
    hold off
    exportgraphics(f, [path, num2str(i), '.png'])
end
% figure %immature
% subplot(3,2,1) %HSC
% plot(t, c(:,1),'LineWidth',2)
% xlabel('Time [days]','FontWeight','bold');
% ylabel('HSC [/kg]','FontWeight','bold');
% ylim([0,5e5]);
% subplot(3,2,2) %MPP
% plot(t, c(:,2),'LineWidth',2)
% xlabel('Time [days]','FontWeight','bold');
% ylabel('MPP [/kg]','FontWeight','bold');
% ylim([0,1e6]);
% subplot(3,2,3) %CMP
% plot(t, c(:,3),'LineWidth',2)
% xlabel('Time [days]','FontWeight','bold');
% ylabel('CMP [/kg]','FontWeight','bold');
% subplot(3,2,4) %MEP
% plot(t, c(:,4),'LineWidth',2)
% xlabel('Time [days]','FontWeight','bold');
% ylabel('MEP [/kg]','FontWeight','bold');
% subplot(3,2,5) %MKb
% plot(t, c(:,5),'LineWidth',2)
% xlabel('Time [days]','FontWeight','bold');
% ylabel('MKb [/kg]','FontWeight','bold');
% subplot(3,2,6) %MK
% plot(t, c(:,6),'LineWidth',2)
% xlabel('Time [days]','FontWeight','bold');
% ylabel('MK [/kg]','FontWeight','bold');