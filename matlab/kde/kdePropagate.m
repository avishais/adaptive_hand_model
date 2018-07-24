function x_next = kdePropagate(x, a, r, data, I) %[X, P]

N = 10;
X = zeros(N,I.state_dim);
P = zeros(N,1);

% [idx, d] = knnsearch(data(:,[I.state_inx I.action_inx]), [x, a], 'K', 100); 
idx = rangesearch(data(:,[I.state_inx I.action_inx]), [x a], 0.01); idx = idx{1};
data_nn = data(idx,:);

X = x+randsphere(N, I.state_dim, r);
for i = 1:N
%     X(i,:) = cirrdnPJ(x, r);
    P(i) = kde([x a X(i,:)], data_nn, I);
end

% P = P/sum(P);

% scatter3(X(:,1),X(:,2),P, 20, P,'filled'); colormap(jet); colorbar;
% view(2)
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
    

% Sample in a cirlce
function y = cirrdnPJ(x, rc)
a = 2*pi*rand;
r = sqrt(rand);
y = (rc*r) * [cos(a) sin(a)] + x;
end

% Sample from distribution
function x = sampleWeight(P, X)
P = P/sum(P);
S = cumsum(P);

d = rand();
j = min(find((d>S)==0));

x = X(j,:);

end

% Sample in a hyper-sphere
function X = randsphere(m,n,r)
 
% This function returns an m by n array, X, in which 
% each of the m rows has the n Cartesian coordinates 
% of a random point uniformly-distributed over the 
% interior of an n-dimensional hypersphere with 
% radius r and center at the origin.  The function 
% 'randn' is initially used to generate m sets of n 
% random variables with independent multivariate 
% normal distribution, with mean 0 and variance 1.
% Then the incomplete gamma function, 'gammainc', 
% is used to map these points radially to fit in the 
% hypersphere of finite radius r with a uniform % spatial distribution.
% Roger Stafford - 12/23/05
 
X = randn(m,n);
s2 = sum(X.^2,2);
X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);

end

