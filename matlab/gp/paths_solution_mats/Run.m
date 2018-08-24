clear all

K = [1 2 5 8];

CumSumAll = [];
L = cell(length(K),1);
for k = 1:length(K)
    
load(['pred_20_' num2str(K(k)) '_1.mat']);

[total_score, CumSum, d] = compare_paths(Sr, S, I);

disp(total_score);

CumSumAll = [CumSumAll CumSum];

L{k} = ['feature conf. ' num2str(K(k))];
end

figure(1)
clf
hold on
plot(d, CumSumAll,'linewidth',3);
hold off
legend(L,'location','northwest','fontsize',14);
xlabel('Traversed path','fontsize',16);
ylabel('Mean-Squared Error','fontsize',16);
xlim([0 d(end)]);

print(['cumsum_test_20_1.png'],'-dpng','-r150');