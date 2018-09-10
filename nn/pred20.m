clear all

px2mm = 0.2621;

%%
data_source = 20;
files = dir(fullfile('./models/', ['net_' num2str(data_source) '_*.netxt']));
files = struct2cell(files)';
files = sortrows(files, 1);

n = size(files,1);
%%

for j = 1:n
    
    mode = files{j,1}(8);
    
    file = ['../data/Ca_' num2str(data_source) '_' num2str(mode)];
    D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest1','Xtest2', 'Xtest3');
    Q = D.Q;
    
    action_inx = Q{1}.action_inx;
    state_inx = Q{1}.state_inx;
    state_nxt_inx = Q{1}.state_nxt_inx;
    
    P = D.Xtest3.data;
    P = P(1:end,:);
    
    [W, b, x_max, x_min, activation] = net_rep(mode, data_source);
    
    %% Open loop
    Sr = P(:,state_inx);
    k = size(Sr,1);
    
    % Sample from distribution
    s = Sr(1,:);
    S = zeros(k, length(state_inx));
    S(1,:) = s;
    for i = 1:size(P,1)-1
        a = P(i, action_inx);
        s = s + Net([s a], W, b, x_max, x_min, activation);
        S(i+1,:) = s;
    end
    
    
    %
    figure(1)
    clf
    plot(Sr(:,1),Sr(:,2),'o-b');
    hold on
    plot(S(:,1),S(:,2),'x-m');
    hold off
    axis equal
    title('open loop');
    
    [total_score{j} max_err{j}] = MSE(Sr, S);
    d{j} = pathLength(Sr*px2mm);
    disp([total_score{j}(end)*px2mm max_err{j}*px2mm]);
    Modes(j) = mode;
    
end

%%
K = [1 2 3 4 5 8 7];

figure(2)
clf
hold on
for i = 1:length(K)
    plot(d{K(i)}, total_score{K(i)}*px2mm,'linewidth',3);
    L{i} = ['feature conf. ' num2str(i)];
end
hold off
set(gca, 'fontsize',16);
legend(L,'location','northwest','fontsize',17);
xlabel('Traversed path (mm)','fontsize',20);
ylabel('RMSE (mm)','fontsize',20);
xlim([0 d{2}(end)]);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.74, 0.4]);

% print(['cumsum_NN_20_1.png'],'-dpng','-r150');

%%

function [cd, max_err] = MSE(S1, S2)

d = zeros(size(S1,1),1);
for i = 1:length(d)
    d(i) = norm(S1(i,1:2)-S2(i,1:2))^2;
end

cd = cumsum(d);

cd = cd ./ (1:length(cd))';

cd = sqrt(cd);

max_err = max(sqrt(d));

end

function d = pathLength(S)

Ssm = MovingAvgFilter(S(:,1:2));

d = zeros(size(Ssm,1),1);
for i = 2:size(Ssm,1)
    d(i) = d(i-1) + norm(Ssm(i,:)-Ssm(i-1,:));    
end

end

function y = MovingAvgFilter(x, windowSize)

if nargin==1
    windowSize = 13;
end

y = x;

w = floor(windowSize/2);
for j = 1:size(x,2)
    for i = w+1:size(x,1)-w-1
        
        y(i,j) = sum(x(i-w:i+w,j))/length(i-w:i+w);
        
    end
    
end
end
