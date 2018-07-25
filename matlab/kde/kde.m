function p = kde(x, data, I)

n = size(data, 1);

p_n = 0;
p_d = 0;
sigma2 = 0.007;
for i = 1:n
    p_n = p_n + GaussianKernal(data(i,:), x, sigma2);
    p_d = p_d + GaussianKernal(data(i,[I.state_inx I.action_inx]), x([I.state_inx I.action_inx]), sigma2);
end

p = p_n / p_d;
% [p_n p_d p]
end

function k = GaussianKernal(u, v, sigma2)

if nargin == 2
    sigma2 = 0.000005;
end

k = exp( -((u-v)*(u-v)')/(2*sigma2) );

end

function k = TriangularKernal(u, v)

d = norm(u,v);
if d > 1
    k = 0;
else
    k = 1 - d;
end

end

