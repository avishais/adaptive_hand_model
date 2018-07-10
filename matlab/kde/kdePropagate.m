function [X, P] = kdePropagate(x, a, r, data) 

N = 10;
X = zeros(N,2);
P = zeros(N,1);

% idx = knnsearch(data(:,1:2), x);
idx = rangesearch(data(:,1:4), [x a], 0.25);
data_nn = data(idx{1},:);

for i = 1:N
    X(i,:) = cirrdnPJ(x, r);
    P(i) = kde([x a X(i,:)], data_nn);
end

% P = P/sum(P);

scatter3(X(:,1),X(:,2),P, 20, P,'filled'); colormap(jet); colorbar;
view(2)
% grid
% hold on
% plot3(x(1),x(2),0.01,'xr')
% hold off

% Sample from distribution
x_next = sampleWeight(P, X);

% Take most probapble state
% [~,i] = max(P);
% x_next = X(i,:);


% plot(x(1),x(2),'xr');
% hold on
% plot(X(:,1),X(:,2),'.');
% th = 0:pi/50:2*pi;
% plot(x(1)+r*cos(th), x(2)+r*sin(th),'-m');
% hold off
% axis equal

end
    

function y = cirrdnPJ(x, rc)
%the function, must be on a folder in matlab path
a = 2*pi*rand;
r = sqrt(rand);
y = (rc*r) * [cos(a) sin(a)] + x;
end

function x = sampleWeight(P, X)
P = P/sum(P);
S = cumsum(P);

d = rand();
j = min(find((d>S)==0));

x = X(j,:);

end

