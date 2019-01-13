clear all

% D = dlmread('../../data/sim_transition_data_cont.db');
D = dlmread('/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont_0.db');

action_inx = 5:6;
state_inx = 1:4;
state_nxt_inx = 7:10;

P = D(250:750,:);

[W, b, x_max, x_min, activation] = net_rep();

%% 

tc = 101;
a = P(tc, action_inx);
x = P(tc, state_inx);
x_next = P(tc,state_nxt_inx);

x_next_pred = x + Net([x a], W, b, x_max, x_min, activation);


%% Open loop

s = P(1,state_inx);
S = s;
Sr = [];%s;
for i = 1:size(P,1)
    a = P(i, action_inx);
    Sr = [Sr; P(i, state_inx)];
    s = s + Net([s a], W, b, x_max, x_min, activation);
    s_real = P(i, state_nxt_inx);
    S = [S; s];    
end

%
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
    sp = s + Net([s a], W, b, x_max, x_min, activation);
    sr = P(i, state_nxt_inx);
    plot([s(1) sr(1)],[s(2) sr(2)],'.-b');
    plot([s(1) sp(1)],[s(2) sp(2)],'.-m');
end
hold off
axis equal
title('closed loop');