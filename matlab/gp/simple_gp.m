clear all

mode = 8;
file = ['../../data/data_25_' num2str(mode)];

D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest');
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

% j_min = 38; j_max = 48;
j_min = 1; j_max = size(Xtest, 1);
Sr = Xtest(j_min:j_max,:);

clear Q
%%
tc = 1000;
a = Xtraining(tc, I.action_inx);
x = Xtraining(tc, I.state_inx);
x_next = Xtraining(tc,I.state_nxt_inx);

x_next_pred = prediction(Xtraining, x, a, I, 1);

%% 

s = Sr(1,I.state_inx);
S = s;
for i = 1:size(Sr,1)
    a = Sr(i, I.action_inx);
    s = prediction(Xtraining, s, a, I);
    S = [S; s];
end

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
title('open loop');

%% Closed loop


sum = 0;
Sp = zeros(size(Sr,1),I.state_dim);
for i = 1:size(Sr,1)
    s = Sr(i,I.state_inx);
    a = Sr(i, I.action_inx);
    sr = Sr(i,I.state_nxt_inx);
    sp = prediction(Xtraining, s, a, I);
    Sp(i,:) = sp;
    sum = sum + norm(sp(1:2)-sr(1:2))^2;
%     drawnow;
end

figure(2)
clf
hold on
plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');
for i = 1:size(Sr,1)
        plot([Sr(i,1) Sp(i,1)],[Sr(i,2) Sp(i,2)],'.-');
end
plot(Sr(1,1),Sr(1,2),'or','markerfacecolor','m');
hold off
axis equal
legend('original path');
title('Closed loop');
Mse = sqrt(sum);
disp(['Error: ' num2str(Mse)]);