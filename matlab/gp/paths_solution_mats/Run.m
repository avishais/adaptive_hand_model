clear all

K = [1 2 3 4 5 8 7];

px2mm = 0.2621;

CumSumAll = [];
L = cell(length(K),1);
for k = 1:length(K)
    
load(['pred_20_' num2str(K(k)) '_1.mat']);
% load(['pred_all_30_3_' num2str(K(k)) '.mat']);

[total_score, CumSum, d, max_error] = compare_paths(Sr(:,1:2)*px2mm, S*px2mm, I);

disp([total_score max_error]);

CumSumAll = [CumSumAll CumSum];

L{k} = ['feature conf. ' num2str(k)];
end

figure(1)
clf
hold on
plot(d, CumSumAll,'linewidth',3);
hold off
set(gca, 'fontsize',12);
legend(L,'location','northwest','fontsize',14);
xlabel('Traversed path (mm)','fontsize',17);
ylabel('RMSE (mm)','fontsize',17);
xlim([0 d(end)]);

% print(['cumsum_test_20_1.png'],'-dpng','-r150');