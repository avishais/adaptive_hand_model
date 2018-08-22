function s_next = gdPropagate(s, a, kdtree, data, I)

[mu, sigma] = prediction(kdtree, data, s, a, I, 1);

s_next = zeros(1, length(mu));
for i = 1:length(mu)
    s_next(i) = normrnd(mu(i),sigma(i));
end

end