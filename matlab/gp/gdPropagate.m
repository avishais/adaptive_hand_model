function s_next = gdPropagate(s, a, data, I)

[mu, sigma] = prediction(data, s, a, I);

s_next = zeros(1, length(mu));
for i = 1:length(mu)
    s_next(i) = normrnd(mu(i),sigma(i));
end

end