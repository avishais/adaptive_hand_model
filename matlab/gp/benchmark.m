clear all

ix = 1:8;

F = cell(length(ix),1);
L = cell(length(ix),1);
for mode = ix
    
    simple_gp;
    
    F{mode} = S;
    L{mode} = loss;
   
end

save(['gpBM3.mat']);

%%
Legend = cell(length(ix)+1,1);
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'--b','linewidth',3,'markerfacecolor','k');
Legend{1} = 'Ground truth';
hold on
for i = 1:8%[1 5 8]
    plot(F{i}(:,1),F{i}(:,2),'linewidth',3);
    Legend{i+1} = num2str(i);
end
hold off
axis equal
legend(Legend);


% drawnow;
% print('imBM.png','-dpng','-r150');

