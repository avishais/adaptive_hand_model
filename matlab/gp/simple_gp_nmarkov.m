clear all
clf

mode = 2;
windowSize = 3;
% file = ['../../data/data_25_' num2str(mode)];
file = ['../../data/data_25_' num2str(mode) '_w' num2str(windowSize)];

Q = load([file '.mat'], 'Q');
Q = Q.Q;
% 
% Xtraining = [];
% for i = 1:size(Q,1)
%     Xtraining = [Xtraining; Q{i}.data(:,1:6)];    
% end

% Xtraining = load('../../data/toyData.db');
Xtraining = load([file '.db']);

% Xtraining = Xtraining(1:300,:);

xmax = max(Xtraining); xmax([1, 5]) = max(xmax([1, 5])); xmax([2, 6]) = max(xmax([2, 6]));
xmin = min(Xtraining); xmin([1, 5]) = min(xmin([1, 5])); xmin([2, 6]) = min(xmin([2, 6]));
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);

% Xtest = Q{20}.data(2:end-1,:);
Xtest = Xtraining(end-3000:end-2000,:);
% Xtest = load('../../data/toyDataPath.db');
% Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

clear Q
%%
tc = 1000;
a = Xtraining(tc, I.action_inx);
x = Xtraining(tc, I.state_inx);
x_next = Xtraining(tc,I.state_nxt_inx);

x_next_pred = prediction(Xtraining, x, a, I, 3);

%% 

% j_min = 38; j_max = 48;
j_min = 1; j_max = 250;%size(Xtest, 1);
Sr = Xtest(j_min:j_max,:);

% Open loop
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');

s = Sr(1,I.state_inx);
S = s(end-1:end);
for i = 1:size(Sr,1)
    a = Sr(i, I.action_inx);
    sp = prediction(Xtraining, s, a, I);
    s = [s(5:end) a sp];
    S = [S; sp];
end
hold on
plot(S(:,1),S(:,2),'.-');
plot(S(1,end-1),S(1,end),'or','markerfacecolor','r');
hold off
axis equal
legend('original path');
title('open loop');

%% Closed loop
figure(3)
clf
hold on
plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');

for i = 1:size(Sr,1)
    s = Sr(i,I.state_inx);
    a = Sr(i, I.action_inx);
    sp = prediction(Xtraining, s, a, I);
    plot([s(1) sp(1)],[s(2) sp(2)],'.-');
%     drawnow;
end
plot(Sr(1,1),Sr(1,2),'or','markerfacecolor','m');
hold off
axis equal
legend('original path');
title('Closed loop');