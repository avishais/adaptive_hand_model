clear all
clc

%% Add EGP distribution

load('traj_dist_20.mat');

K = [1:35 37:41 43:49 51:75];

Sx = []; Sy = [];
for i = K
    Sx = [Sx S{i}(:,1)];
    Sy = [Sy S{i}(:,2)];
end

Sm = [mean(Sx')' mean(Sy')'];
Ss = [std(Sx')' std(Sy')'];
Sx = [Sm(:,1)-Ss(:,1) Sm(:,1)+Ss(:,1)];
Sy = [Sm(:,2)-Ss(:,2) Sm(:,2)+Ss(:,2)];

Sx_egp = Sx;
Sy_egp = Sy;
Sm_egp = Sm;
Tegp = data{1}.T;


%% With NN
load('traj_dist_20_dm.mat');

load('../../../nn/NN_dist_20.mat');

px2mm = 0.2621;

U = repmat(I.base_pos*px2mm, data{1}.n, 1);
Ut = repmat(I.base_pos*px2mm, data{it}.n, 1);
Unn = repmat(I.base_pos*px2mm, length(NNx), 1);

figure(2)
clf
subplot(211)
hold on
fill([Tegp; flipud(Tegp)], [Sx_egp(:,1)*px2mm+U(:,1); flipud(Sx_egp(:,2)*px2mm+U(:,1))],'m');
fill([data{1}.T; flipud(data{1}.T)], [Sx(:,1)*px2mm+U(:,1); flipud(Sx(:,2)*px2mm+U(:,1))],'y');
fill([T_nn; flipud(T_nn)], [NNx(:,1)*px2mm+Unn(:,1); flipud(NNx(:,2)*px2mm+Unn(:,1))],'g');
fill([data{it}.T; flipud(data{it}.T)], [Rx(:,1)*px2mm+Ut(:,1); flipud(Rx(:,2)*px2mm+Ut(:,1))],'-b');
% plot(Tegp,Sm_egp(:,1)*px2mm+U(:,1),'-.k','linewidth',2.5);
plot(data{1}.T,Sm(:,1)*px2mm+U(:,1),':k','linewidth',2.5);
plot(T_nn,NNm(:,1)*px2mm+Unn(:,1),'--k','linewidth',2.5);
plot(data{it}.T,Rm(:,1)*px2mm+Ut(:,1),'-k','linewidth',2.5);
% plot([data{1}.T; flipud(data{1}.T)], [Sx(:,1); flipud(Sx(:,2))],'-k','linewidth',4);
hold off
set(gca, 'FontSize', 12)
xlabel('Time (sec)','fontsize', 17);
ylabel('Position - x axis (mm)','fontsize', 17);
% legend({'pred. std.','ground truth std.','pred. mean','ground truth mean'},'location','northwest','fontsize',14);
xlim([0 max(data{it}.T)-2]);
ylim([130 180])

subplot(212)
hold on
fill([Tegp; flipud(Tegp)], [Sy_egp(:,1)*px2mm+U(:,2); flipud(Sy_egp(:,2)*px2mm+U(:,2))],'m');
fill([data{1}.T; flipud(data{1}.T)], [Sy(:,1)*px2mm+U(:,2); flipud(Sy(:,2)*px2mm+U(:,2))],'y');
fill([T_nn; flipud(T_nn)], [NNy(:,1)*px2mm+Unn(:,2); flipud(NNy(:,2)*px2mm+Unn(:,2))],'g');
fill([data{it}.T; flipud(data{it}.T)], [Ry(:,1)*px2mm+Ut(:,2); flipud(Ry(:,2)*px2mm+Ut(:,2))],'b');
% plot(Tegp,Sm_egp(:,2)*px2mm+U(:,2),'-.k','linewidth',2.5);
plot(data{1}.T,Sm(:,2)*px2mm+U(:,2),':k','linewidth',2.5);
plot(T_nn,NNm(:,2)*px2mm+Unn(:,2),'--k','linewidth',2.5);
plot(data{it}.T,Rm(:,2)*px2mm+Ut(:,2),'-k','linewidth',2.5);
% plot([data{1}.T; flipud(data{1}.T)], [Sy(:,1); flipud(Sy(:,2))],'-k','linewidth',3);
hold off
set(gca, 'FontSize', 12)
xlabel('Time (sec)','fontsize', 17);
ylabel('Position - y axis (mm)','fontsize', 17);
legend({'EGP pred. std.','MLGP pred. std.','NN std.','ground truth std.','MLGP pred. mean','NN mean','ground truth mean'},'location','northwest','fontsize',14);
xlim([0 max(data{it}.T)-2]);
ylim([25 65])






%%

% print(['distributions20withNN.png'],'-dpng','-r150');