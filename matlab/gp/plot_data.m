clear all
close all

mode = 2;
file = ['../../data/data_25_' num2str(mode)];
X = load([file '.db']);

xmax = max(X); xmax([1, 5]) = max(xmax([1, 5])); xmax([2, 6]) = max(xmax([2, 6]));
xmin = min(X); xmin([1, 5]) = min(xmin([1, 5])); xmin([2, 6]) = min(xmin([2, 6]));
X = (X-repmat(xmin, size(X,1), 1))./repmat(xmax-xmin, size(X,1), 1);

openfig('cl.fig')

i = 77735+1;
x = X(i,:);

[idx, ~] = knnsearch(X(:,1:4), x(1:4), 'K', 100);
% [idx, ~] = knnsearch(X(:,1:2), x(1:2), 'K', 100);

data_nn = X(idx,:);

%%
hold on
plot(data_nn(:,1),data_nn(:,2),'.r')
for i = 1:size(data_nn,1)
    plot(data_nn(i,[1 5]),data_nn(i,[2 6]),'.-m');
    d = what_action(data_nn(i,[3 4]));
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