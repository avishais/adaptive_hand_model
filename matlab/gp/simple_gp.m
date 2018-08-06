% clear all
% warning('off','all')


ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

% mode = 5;
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
% Xtest =  load('../../data/c_25_7_rerun_processed.txt');
% Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

% j_min = 250; j_max = 350;%size(Xtest, 1);
j_min = 1; j_max = size(Xtest, 1);
Sr = Xtest(j_min:j_max,:);

global W
% W = diag([10 10 2 2 1 1]);
W = diag([ones(1,2)*3 ones(1,I.state_dim)]);

kdtree = createns(Xtraining(:,[I.state_inx I.action_inx]),'Distance',@distfun);

clear Q D
%%
% tc = 1000;
% a = Xtraining(tc, I.action_inx);
% x = Xtraining(tc, I.state_inx);
% x_next = Xtraining(tc,I.state_nxt_inx);
% 
% a = [1 1];
% x = [0.5605 0.1456];
% % x_next = [0.5603 0.1463];
% x_next_pred = prediction(Xtraining, x, a, I, 1)

%% open loop
tic;
s = Sr(1,I.state_inx);
S = zeros(size(Sr,1), I.state_dim);
S(1,:) = s;
loss = 0;
for i = 1:size(Sr,1)-1
    disp(['Step: ' num2str(i)]);
    a = Sr(i, I.action_inx);
    [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);
    S(i+1,:) = s;
    loss = loss + norm(s - Sr(i+1, I.state_nxt_inx));
end

loss = loss / size(Sr,1);
disp(toc)
%% Closed loop

% sum = 0;
% Sp = zeros(size(Sr,1),I.state_dim);
% for i = 1:size(Sr,1)
%     s = Sr(i,I.state_inx);
%     a = Sr(i, I.action_inx);
%     sr = Sr(i,I.state_nxt_inx);
%     sp = prediction(Xtraining, s, a, I);
%     Sp(i,:) = sp;
%     sum = sum + norm(sp(1:2)-sr(1:2))^2;
% %     drawnow;
% end

%%

figure(1)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on
plot(S(:,1),S(:,2),'.-r');
plot(S(1,1),S(1,2),'or','markerfacecolor','r');
hold off
axis equal
legend('ground truth','predicted path');
title(['open loop - ' num2str(mode) ', MSE: ' num2str(loss)]);
disp(['Loss: ' num2str(loss)]);

% figure(2)
% clf
% hold on
% plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');
% for i = 1:size(Sr,1)
%         plot([Sr(i,1) Sp(i,1)],[Sr(i,2) Sp(i,2)],'.-');
% end
% plot(Sr(1,1),Sr(1,2),'or','markerfacecolor','m');
% hold off
% axis equal
% legend('original path');
% title('Closed loop');
% Mse = sqrt(sum);
% disp(['Error: ' num2str(Mse)]);

