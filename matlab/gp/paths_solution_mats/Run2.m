clear all
clc

K = [5 8];%[1 2 3 4 5 8 7];
data_sources = {'26_1','30_3','36_2'};
objs = {'Butter can','Glue stick','Hair-spary'};
realFC = [5 6]; %[1 2 3 4 5 6 7];

px2mm = 0.2621;

% CumSumAll = [];
L = cell(length(K)*length(data_sources),1);
j = 1;
for i = 1:length(data_sources)
    
    for k = 1:length(K)
        
        
        load(['pred_all_' data_sources{i}  '_' num2str(K(k)) '.mat'] );
        
        [total_score, CumSum, d] = compare_paths(Sr(:,1:2)*px2mm, S*px2mm, I);
        
        disp(total_score);
        
        CumSumAll{j} = CumSum;
        D{j} = d;
        
        L{j} = [objs{i} ', feature conf. ' num2str(realFC(k))];
        
        j = j + 1;
    end
end

%%

figure(1)
clf
hold on
for i = 1:length(CumSumAll)
    plot(D{i}, CumSumAll{i},'linewidth',3);
end
hold off
set(gca, 'fontsize',12);
legend(L,'location','northwest','fontsize',14);
xlabel('Traversed path (mm)','fontsize',17);
ylabel('RMSE (mm)','fontsize',17);
xlim([0 d(end)]);

% print(['cumsum_test_20_1.png'],'-dpng','-r150');