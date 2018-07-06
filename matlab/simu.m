clear all

Q = load('../data/data1.mat', 'Q');
Q = Q.Q;

P = Q{2}.data;
P(1,:) = [];

%%
[W, b, x_max, x_min] = net_rep;
s = P(1,1:2);
S = s;
Sr = s;
for i = 1:size(P,1)
    a = P(i,3:4);
    Sr = [Sr; P(i,1:2)];
    s = Net([s a], W, b, x_max, x_min);
    s_real = P(i,5:6);
    S = [S; s];    
end

%%
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'.-b');
hold on
plot(S(:,1),S(:,2),'.-m');
hold off