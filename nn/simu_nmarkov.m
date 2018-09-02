clear all
num_net = 2;
windowSize = 10;
Q = load(['../data/data_25_' num2str(num_net) '_w' num2str(windowSize) '.mat'], 'Q');
Q = Q.Q;
action_inx = Q{1}.action_inx;
state_inx = Q{1}.state_inx;
state_nxt_inx = Q{1}.state_nxt_inx;

P = load(['../data/data_25_' num2str(num_net) '_w' num2str(windowSize) '.db']);
P = P(end-3000:end-2000,:);
% P = P(end-2000:end,:);

% P = Q{7}.data;
% P(1,:) = [];

% P = load('../data/toyDataPath.db');
% action_inx = 3:4;
% state_inx = 1:2;
% state_nxt_inx = 5:6;

[W, b, x_max, x_min, activation] = net_rep(num_net, windowSize);

%% Open loop

s = P(1,state_inx);
S = s(end-1:end);
Sr = [];%s;
for i = 1:size(P,1)
    a = P(i, action_inx);  
    Sr = [Sr; P(i, state_inx(end-1:end))];
    s_next = s(end-1:end) + Net([s a], W, b, x_max, x_min, activation);
    s = [s(5:end) a s_next];
    S = [S; s_next];    
end

%%
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'.-b');
hold on
plot(S(:,1),S(:,2),'.-m');
hold off
axis equal
title('open loop');

%% Closed loop

figure(2)
clf
hold on

for i = 1:size(P,1)
    s = P(i,state_inx);
    a = P(i, action_inx);
    sp = s(end-1:end) + Net([s a], W, b, x_max, x_min, activation);
    sr = P(i, state_nxt_inx);
    plot([s(1) sr(1)],[s(2) sr(2)],'.-b');
    plot([s(1) sp(1)],[s(2) sp(2)],'.-m');
end
hold off
axis equal
title('closed loop');