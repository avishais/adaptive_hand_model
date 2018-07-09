function x_next = kdePropagate(x, a, r, data) 

N = 100;
X = zeros(N,2);
P = zeros(N,1);

% idx = knnsearch(data(:,1:4), [x a]);
idx = rangesearch(data(:,1:4),[x a],0.04);
data_nn = data(idx{1},:);
size(data_nn,1)

for i = 1:N
    X(i,:) = cirrdnPJ(x, r);
    P(i) = kde([x a X(i,:)], data_nn);
end

P = P/sum(P);

% Fit data to gaussian - may not be the best solution
mu = mean(P);
sigma = std(P);
p = normrnd(mu, sigma);

% Sample from distribution
[~,i] = min(abs(P-p));
x_next = X(i,:);

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

