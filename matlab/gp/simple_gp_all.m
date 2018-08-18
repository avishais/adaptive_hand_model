warning('off','all')

if ~exist('is_nm','var')
    clear all
    
    mode = 8;
end

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 4;
w = [1 1 1 1 1 1 1];
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, 'all');

Sr = Xtest;

%% open loop
figure(2)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on

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
    
    if ~mod(i, 20)
        plot(S(1:i,1),S(1:i,2),'.-r');
        drawnow;
    end
end
S = S(1:i+1,:);
hold off

loss = loss / size(Sr,1);
disp(toc)

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

%% Closed loop

figure(2)
clf
hold on
plot(Sr(:,1),Sr(:,2),'o-b','linewidth',2,'markerfacecolor','k');

Sp = zeros(size(Sr,1),I.state_dim);
for i = 1:size(Sr,1)
    disp(['Step: ' num2str(i)]);
    
    s = Sr(i,I.state_inx);
    a = Sr(i, I.action_inx);
    sr = Sr(i,I.state_nxt_inx);
    sp = prediction(kdtree, Xtraining, s, a, I);
    Sp(i,:) = sp;
%     drawnow;

    if ~mod(i, 10)
        for j = 1:i
            plot([Sr(j,1) Sp(j,1)],[Sr(j,2) Sp(j,2)],'.-r');
        end
        drawnow;
    end
end

hold off

%%
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

