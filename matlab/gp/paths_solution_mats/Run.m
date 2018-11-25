clear all

K = [1 2 3 4 5 8 7];

px2mm = 0.2621;

CumSumAll = [];
L = cell(length(K),1);
for k = 1:length(K)
    
load(['pred_20_' num2str(K(k)) '_2.mat']);
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
set(gca, 'fontsize',14);
legend(L,'location','northwest','fontsize',17);
xlabel('Traversed path (mm)','fontsize',20);
ylabel('RMSE (mm)','fontsize',20);
xlim([0 d(end)]);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.74, 0.6]);

% print(['cumsum_test_20_1.png'],'-dpng','-r150');