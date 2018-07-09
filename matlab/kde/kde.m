function p = kde(x, data)

n = max(size(data));

p_n = 0;
p_d = 0;
for i = 1:n
    p_n = p_n + GaussianKernal(data(i,:), x);
    p_d = p_d + GaussianKernal(data(i,1:4),x(1:4));
end

p = p_n / p_d;

end

function k = GaussianKernal(u, v)

sigma2 = 1;
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

