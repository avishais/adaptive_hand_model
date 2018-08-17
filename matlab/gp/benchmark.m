clear all

ix = [1 5 8 9 10];

F = cell(length(ix),1);
L = cell(length(ix),1);
for mode = ix
    disp(mode);
    
    simple_gp_all;
    
    F{mode} = S;
    L{mode} = loss;
   
end

save(['gpBMall.mat']);

%%
clear all
load(['gpBMall.mat']);

% Legend = cell(length(ix)+1,1);
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'--b','linewidth',3,'markerfacecolor','k');
Legend{1} = 'Ground truth';
hold on
j = 2;
for i = [1 5 8]
    plot(F{i}(:,1),F{i}(:,2),'linewidth',3);
    Legend{j} = num2str(i);
    j = j + 1;
end
hold off
axis equal
legend(Legend);


% drawnow;
% print('imBM.png','-dpng','-r150');

