function D2 = distfun(ZI,ZJ)

global W

% W = diag([3 3 1 1 1 1]);
if isempty(W)
    W = diag(ones(1,size(ZI,2)));
end    

n = size(ZJ,1);
D2 = zeros(n,1);
for i = 1:n
    Z = ZI-ZJ(i,:);
    D2(i) = Z*W*Z';
end
    
end