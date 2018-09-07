clear all

px2mm = 0.2621;

load(['pred_20_8_1.mat']);
[total_score1, CumSum1, d1] = compare_paths(Sr(:,1:2)*px2mm, S*px2mm, I);

load(['pred_20_8_1_dm.mat']);
[total_score2, CumSum2, d2] = compare_paths(Sr(:,1:2)*px2mm, S*px2mm, I);


figure(1)
clf
hold on
plot(d1, CumSum1,'r','linewidth',3);
plot(d2, CumSum2,'--b','linewidth',3);
hold off
set(gca, 'fontsize',12);
% legend(L,'location','northwest','fontsize',14);
xlabel('Traversed path (mm)','fontsize',17);
ylabel('RMSE (mm)','fontsize',17);
xlim([0 d1(end)]);

