clear all
D = load('data_25_1.mat');
Q = D.Q;

% for i = 65%1:size(Q,1)
%     i
%     Sr = Q{i}.data;
%     plot(Sr(340:end-350,1),Sr(340:end-350,2),'-b','linewidth',3,'markerfacecolor','k');
%     axis equal
%     i;
% end

%%

Sr = D.Xtest;
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');