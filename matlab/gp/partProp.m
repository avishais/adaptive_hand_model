clear all
warning('off','all')


ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

UseToyData = false;

mode = 1;
file = ['../../data/Ca_25_' num2str(mode)];

D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest','Xtest2');
Q = D.Q;
I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

if UseToyData
    Xtraining = load('../../data/toyData.db');
    Xtest = load('../../data/toyDataPath.db');
else
    Xtraining = D.Xtraining; 
    Xtest = D.Xtest;
end

xmax = max(Xtraining); 
xmin = min(Xtraining); 
for i = 1:I.state_dim
    id = [i i+I.state_dim+length(I.action_inx)];
    xmax(id) = max(xmax(id));
    xmin(id) = min(xmin(id));
end
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);
Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

j_min = 270; j_max = 600;%size(Xtest, 1);
Sr = Xtest(j_min:j_max,:);

global W
% W = diag([10 10 2 2 1 1]);
W = diag([ones(1,2)*1 ones(1,I.state_dim)]);

kdtree = createns(Xtraining(:,[I.state_inx I.action_inx]),'Distance',@distfun);

clear Q D

%%
Np = 10;

s = Sr(1, I.state_inx);
P_prev = repmat(s, Np, 1);

figure(1)
clf;
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
axis equal

hold on

S = zeros(size(Sr,1)-1, I.state_dim);
S(1,:) = mean(P_prev);
for i = 1:size(S,1)-1
    disp(['Step: ', num2str(i)]);
    
    a = Sr(i, I.action_inx);
    
    P = zeros(Np, I.state_dim);
    for k = 1:Np
        s = P_prev(k,:);
        [mu, sigma] = prediction(kdtree, Xtraining, s, a, I, 1);
        for j = 1:I.state_dim
            P(k,j) = normrnd(mu(j),sigma(j));
        end
    end
    S(i+1,:) = mean(P);
    
    ix = convhull(P(:,1),P(:,2));
    patch(P(ix,1),P(ix,2),'y')
    
    plot(P(:,1),P(:,2),'.k');
    plot(S(1:i,1),S(1:i,2),'-r','markersize',4,'markerfacecolor','r');
    
    P_prev = P;
%     drawnow;
    
end

plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
plot(S(:,1),S(:,2),':r','linewidth',3,'markerfacecolor','r');

hold off



