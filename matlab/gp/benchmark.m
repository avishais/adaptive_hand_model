clear all

is_nm = true;

modes = [5 8 11];

F = cell(length(modes),1);
L = cell(length(modes),1);

for ai = 1:length(modes)
    disp(modes(ai));
    
    mode = modes(ai);
    
    simple_gp_all;
    
    F{ai} = S;
    L{ai} = loss;
   
end

save(['gpBMall-30.mat']);

%%
clear all
load(['gpBMall-30.mat']);

% Legend = cell(length(ix)+1,1);
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'--b','linewidth',3,'markerfacecolor','k');
Legend{1} = 'Ground truth';
hold on
for i = 1:length(modes)-1
    plot(F{i}(:,1),F{i}(:,2),'linewidth',3);
    Legend{i+1} = num2str(modes(i));
end
hold off
axis equal
legend(Legend);


% drawnow;
% print('imBM.png','-dpng','-r150');

is_nm = false;
