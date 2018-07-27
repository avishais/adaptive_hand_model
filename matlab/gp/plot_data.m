clear all
close all

mode = 5;
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

i = 294;
x = Xtest(i,:);

[idx, ~] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), x([I.state_inx I.action_inx]), 'K', 100, 'Distance',@distfun);
% [idx, ~] = knnsearch(X(:,1:2), x(1:2), 'K', 100);

data_nn = Xtraining(idx,:);

%%
openfig('ol.fig');

hold on
plot(data_nn(:,1),data_nn(:,2),'.r')
for i = 1:size(data_nn,1)
    plot(data_nn(i,[I.state_inx(1) I.state_nxt_inx(1)]),data_nn(i,[I.state_inx(2) I.state_nxt_inx(2)]),'.-m');
    d = what_action(data_nn(i,I.action_inx));
    quiver(data_nn(i,1),data_nn(i,2), d(1), d(2),0.002,'k');
end
plot(x(1),x(2),'ob','markersize',10);
hold off
axis([0.99*min(data_nn(:,1)) 1.01*max(data_nn(:,1)) 0.99*min(data_nn(:,2)) 1.01*max(data_nn(:,2))]);
legend off
% axis equal



%%
function d = what_action(a)
if all(a==[0 0])
    d = [0 -1];
else if all(a==[1 1])
        d = [0 1];
    else if all(a==[0 1])
            d = [1 0];
        else
            if all(a==[1 0])
                d = [-1 0];
            end
        end
    end
end
end

function D2 = distfun(ZI,ZJ)

% W = diag([3 3 1 1 1.5 1.5]);
W = diag([20 20 1 1 1 1]);

n = size(ZJ,1);
D2 = zeros(n,1);
for i = 1:n
    Z = ZI-ZJ(i,:);
    D2(i) = Z*W*Z';
end
    
end