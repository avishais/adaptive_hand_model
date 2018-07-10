clear all
num_net = 1;
Q = load(['../data/data' num2str(num_net) '.mat'], 'Q');
Q = Q.Q;
action_inx = Q{1}.action_inx;
state_inx = Q{1}.state_inx;
state_nxt_inx = Q{1}.state_nxt_inx;

P = Q{2}.data;
P(1,:) = [];

%%
[W, b, x_max, x_min] = net_rep(num_net);
s = P(1,state_inx);
S = s;
Sr = [];%s;
for i = 1:10%size(P,1)
    a = P(i, action_inx);
    Sr = [Sr; P(i, state_inx)];
    s = s + Net([s a], W, b, x_max, x_min);
    s_real = P(i, state_nxt_inx);
    S = [S; s];    
end

%%
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'.-b');
hold on
plot(S(:,1),S(:,2),'.-m');
hold off