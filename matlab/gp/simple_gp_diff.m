clear all
clear global W

% ps = parallel.Settings;
% ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.
% parpool;

mode = 5;
file = ['../../data/data_25_' num2str(mode)];

D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest','Xtest2');
Q = D.Q;
I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

% Xtraining = load('../../data/toyData.db');
Xtraining = D.Xtraining;
Xtest = D.Xtest;

Xtraining(:,I.state_nxt_inx) = Xtraining(:,I.state_nxt_inx) - Xtraining(:,I.state_inx);
Xtest(:,I.state_nxt_inx) = Xtest(:,I.state_nxt_inx) - Xtest(:,I.state_inx);

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
W = diag([1 1 0.5 0.5 1 1]);

clear Q D
%%
tc = 1000;
a = Xtraining(tc, I.action_inx);
x = Xtraining(tc, I.state_inx);
x_next = denormz(x, xmin, xmax) + denormz(Xtraining(tc,I.state_nxt_inx), xmin, xmax);

x_next_pred = denormz(x, xmin, xmax) + denormz(prediction(Xtraining, x, a, I, 1), xmin, xmax);

%% 

s = Sr(1,I.state_inx);
S = zeros(size(Sr,1), I.state_dim);
S(1,:) = s;
loss = 0;
for i = 1:size(Sr,1)-1
    a = Sr(i, I.action_inx);
    s = denormz(s, xmin, xmax) + denormz(prediction(Xtraining, s, a, I), xmin, xmax);
    s = normz(s, xmin, xmax);
    S(i+1,:) = s;
    loss = loss + norm(s - Sr(i+1, I.state_nxt_inx));
%     Sr(i,:) = denormz(Sr(i,:), xmin, xmax);
end

% Sr(end,:) = denormz(Sr(end,:), xmin, xmax);
loss = loss / size(Sr,1);
%%
% Open loop
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

%%

function dx = normz(x, xmin, xmax)

n = length(x);
dx = (x - xmin(1:n)) ./ (xmax(1:n)-xmin(1:n));

end

function dx = denormz(x, xmin, xmax)

n = length(x);
dx = x .* (xmax(1:n)-xmin(1:n)) + xmin(1:n);

end