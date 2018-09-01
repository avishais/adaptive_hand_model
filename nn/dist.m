clear all
num_net = 8;
data_source = '20';
file = ['../data/Ca_' data_source '_' num2str(num_net)];
D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest1','Xtest2', 'Xtest3');
Q = D.Q;

action_inx = Q{1}.action_inx;
state_inx = Q{1}.state_inx;
state_nxt_inx = Q{1}.state_nxt_inx;

P = D.Xtest1.data;
P = P(700:end,:);

[W, b, x_max, x_min, activation] = net_rep(num_net);

%%

D = load('../matlab/gp/forPaper/traj_dist_20.mat','start_mean','start_std');
start_mean = D.start_mean;
start_std = D.start_std;
clear D;

%% Open loop
Sr = P(:,state_inx);
k = size(Sr,1);

S = cell(100,1);
for j = 1:length(S)
    
    % Sample from distribution
    s = zeros(1, length(start_mean));
    for ia = 1:length(start_mean)
        s(ia) = normrnd(start_mean(ia),start_std(ia));
    end
    
    S{j} = zeros(k, length(state_inx));
    S{j}(1,:) = s;
    for i = 1:size(P,1)-1
        a = P(i, action_inx);
        s = s + Net([s a], W, b, x_max, x_min, activation);
        S{j}(i+1,:) = s;
    end
    
end

%
figure(1)
clf
plot(Sr(:,1),Sr(:,2),'.-b');
hold on
for j = 1:length(S)
    plot(S{j}(:,1),S{j}(:,2),'.-m');
end
hold off
axis equal
title('open loop');

%%


NNx = []; NNy = [];
for i = 1:length(S)
    NNx = [NNx S{i}(:,1)];
    NNy = [NNy S{i}(:,2)];
end

NNm = [mean(NNx')' mean(NNy')'];
NNs = [std(NNx')' std(NNy')'];
NNx = [NNm(:,1)-NNs(:,1) NNm(:,1)+NNs(:,1)];
NNy = [NNm(:,2)-NNs(:,2) NNm(:,2)+NNs(:,2)];

T_nn = (0:1/15:1/15*(k-1))';
save('NN_dist_20.mat','NNx','NNy','NNm','NNs','T_nn');
