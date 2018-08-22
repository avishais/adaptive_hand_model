clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 1;
mode = 5;
w = 100;
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, '25');

Sr = Xtest;

% Sr(:,5:6) = fliplr(Sr(:,5:6));
%% Point validation
tc = 50;
a = [1 0];%Xtest(tc, I.action_inx);
x = Xtest(tc, I.state_inx);
x_next = Xtest(tc,I.state_nxt_inx);
% 

x_next_pred = prediction(kdtree, Xtraining, x, a, I, 1);

figure(2)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on
axis equal
plot([x(1) x_next_pred(1)],[x(2) x_next_pred(2)],'.-r');
plot(x(1),x(2),'or','markerfacecolor','m');


% s0 = [0.48364569, 0.25166836, 0.74890212, 0.70639912, 0.65672628, 0.86405539];
% S = s0;
% s = s0;
% for i = 1:4
%     s = prediction(kdtree, Xtraining, s, a, I, 1);
%     S = [S; s];
% end


%% open loop
figure(2)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on
axis equal

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
    
    if ~mod(i, 10)
        plot(S(1:i,1),S(1:i,2),'.-r');
        drawnow;
    end
end
S = S(1:i+1,:);
hold off

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

%%

function d = Action(a)
% a = (a+0.06)/0.12;
if all(a==[0 0])
    d = [0 -1];
else if all(a==[1 1])
        d = [0 1];
    else if all(a==[1 0])
            d = [-1 0];
        else
            if all(a==[0 1])
                d = [1 0];
            end
        end
    end
end
end
