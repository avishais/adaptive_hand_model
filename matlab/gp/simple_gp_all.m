% clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 1;
% mode = 1;
w = 1;
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, 'all');

Sr = Xtest;

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
S = S(1:i+1,:);

loss = loss / size(Sr,1);
disp(toc)

%%

% figure(1)
% clf
% plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
% hold on
% plot(S(:,1),S(:,2),'.-r');
% plot(S(1,1),S(1,2),'or','markerfacecolor','r');
% hold off
% axis equal
% legend('ground truth','predicted path');
% title(['open loop - ' num2str(mode) ', MSE: ' num2str(loss)]);
% disp(['Loss: ' num2str(loss)]);
