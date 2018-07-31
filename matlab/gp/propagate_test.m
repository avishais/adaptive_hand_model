clear all

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
poolobj = gcp; % If no pool, do not create new one.

mode = 5;
file = ['../../data/data_25_' num2str(mode)];

D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest','Xtest2');
Q = D.Q;
I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

% Xtraining = load('../../data/toyData.db');
Xtraining = D.Xtraining; %load([file '.db']);
Xtest = D.Xtest;

xmax = max(Xtraining); 
xmin = min(Xtraining); 
for i = 1:I.state_dim
    id = [i i+I.state_dim+length(I.action_inx)];
    xmax(id) = max(xmax(id));
    xmin(id) = min(xmin(id));
end
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);
Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

% Xtest = load('../../data/toyDataPath.db');
% Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

% j_min = 250; j_max = 350;%size(Xtest, 1);
j_min = 1; j_max = size(Xtest, 1);
Sr = Xtest(j_min:j_max,:);

global W
% W = diag([10 10 2 2 1 1]);
W = diag([3 3 1 1 1 1]);

clear Q D

%% Open loop

s = Sr(1,I.state_inx);
loss = 0;
S = zeros(size(Sr,1), I.state_dim);
S(1,:) = s;
for i = 1:size(Sr,1)-1
    a = Sr(i, I.action_inx);
    
    s = gdPropagate(s, a, Xtraining, I);
    
    S(i+1,:) = s;
    loss = loss + norm(s - Sr(i+1, I.state_nxt_inx));
end

loss = loss / size(Sr,1);

%%

figure(1)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on
plot(S(:,1),S(:,2),'.-r');
plot(S(1,1),S(1,2),'or','markerfacecolor','r');
hold off
axis equal
legend('original path');
title(['open loop - ' num2str(mode) ', MSE: ' num2str(loss)]);
disp(['Loss: ' num2str(loss)]);

