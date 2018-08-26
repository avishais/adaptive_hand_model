function [total_score, CumSum, d] = compare_paths(Sr, S, I)

SRI = zeros(size(Sr,1),2);
SI = zeros(size(S,1),2);
for i = 1:size(Sr,1)
    SRI(i,:) = project2image(Sr(i,1:2), I);
    SI(i,:) = project2image(S(i,1:2), I);
end

mse = MSE(SRI,SI);
% mse = dtw(Sr,S);


d = zeros(size(SRI,1),1);
for i = 2:size(SRI,1)
    d(i) = d(i-1) + norm(SRI(i,:)-SRI(i-1,:));    
end

total_score = mse(end);

CumSum = mse;

end

function d = MSE(S1, S2)

d = zeros(size(S1,1),1);
for i = 1:length(d)
    d(i) = norm(S1(i,1:2)-S2(i,1:2))^2;
end

d = cumsum(d);

d = d ./ (1:length(d))';

end

function d = dtw(S1, S2)

d = zeros(size(S1,1),1);
for i = 1:length(d)
    d(i) = DTW(S1(1:i,1:2),S2(1:i,1:2), 0);    
end

end

function sd = denormz(s, I)

xmin = I.xmin(1:length(s));
xmax = I.xmax(1:length(s));

sd = s .* (xmax-xmin) + xmin;

end

function sd = project2image(s, I)

s = denormz(s, I);

R = [cos(I.theta) -sin(I.theta); sin(I.theta) cos(I.theta)];
s = (R' * s')';

sd = s + I.base_pos(1:2);

end