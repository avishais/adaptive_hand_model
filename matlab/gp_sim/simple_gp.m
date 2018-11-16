clear all
warning('off','all')

px2mm = 1;

GP = gp_class(1, false);

Sr = GP.Xtest;
I = GP.I;

%%
% i = 250;
% sa = GP.Xtest(i,[I.state_inx I.action_inx]);
% s_next = GP.normz(GP.Xtest(i, I.state_nxt_inx));
% 
% sa = GP.normz(sa);
% 
% [s, s2] = GP.predict(sa(1:4), sa(5:6))

% s = GP.denormz(s)


%% open loop
figure(2)
clf
hold on
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','y');
axis equal

tic;
s = Sr(1,I.state_inx);
S = zeros(size(Sr,1), I.state_dim);
SI = zeros(size(Sr,1), 2);
S(1,:) = s;
loss = 0;
for i = 1:size(Sr,1)-1
    a = Sr(i, I.action_inx);
    disp(['Step: ' num2str(i) ', action: ' num2str(a)]);
    [s, s2] = GP.predict(s, a);
    S(i+1,:) = s;
        
    if ~mod(i, 10)
        plot(S(1:i,1),S(1:i,2),'.-m');
        drawnow;
        disp(['mse = ' num2str(MSE(Sr(1:i,1:2), S(1:i,1:2)) * px2mm)]);
    end
end
S = S(1:i+1,:);
hold off

disp(toc)

disp(['mse = ' num2str(MSE(Sr, S) * px2mm)]);

% save(['./paths_solution_mats/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_test.mat'],'data_source','I','loss','mode','S','SI','Sr','SRI','test_num','w','Xtest');

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
title(['open loop, MSE: ' num2str(loss)]);
disp(['Loss: ' num2str(loss)]);


%% Functions

function d = MSE(S1, S2)

d = zeros(size(S1,1),1);
for i = 1:length(d)
    d(i) = norm(S1(i,1:2)-S2(i,1:2))^2;
end

d = cumsum(d);

d = d ./ (1:length(d))';

d = sqrt(d);

d = d(end);

end

