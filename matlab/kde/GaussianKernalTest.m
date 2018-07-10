% for i = 1:length(x)
%     y(i) = GaussianKernal(x(i),1);
% end
% plot(x,y)
x = Xtraining(1,:);
data = Xtraining(1:end,:);

a = [1 0];
x = [x(1:2) a cirrdnPJ(x(1:2), 0.0083)];
[idx, d] = rangesearch(data(:,1:2),x(1:2),0.01);
data_nn = data(idx{1},:);
d = d{1}';

D = zeros(size(data_nn,1),4);
for i = 1:size(data_nn,1)
%     D(i,1) = norm(data_nn(i,1:2)-x(1:2))^2;
    D(i,1) = norm(data_nn(i,1:4)-x(1:4))^2;
    D(i,2) = norm(data_nn(i,:)-x)^2;
    D(i,3) = GaussianKernal(data_nn(i,1:4), x(1:4), 0.000005);
    D(i,4) = GaussianKernal(data_nn(i,:), x, 0.05);
    
end

f = kde1(x, data_nn);

s = sum(D);

plot(data_nn(:,1),data_nn(:,2),'o');
hold on
plot(data_nn(:,5),data_nn(:,6),'x');
hold off

function p = kde1(x, data)

n = max(size(data));

p_n = 0;
p_d = 0;
for i = 1:n
    p_n = p_n + GaussianKernal(data(i,:), x, 0.005);
    p_d = p_d + GaussianKernal(data(i,1:4), x(1:4), 0.000007);
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


function k = GaussianKernalN(u)

sigma2 = 0.000005;
k = exp( -(u)/(2*sigma2) );

end

function y = cirrdnPJ(x, rc)
%the function, must be on a folder in matlab path
a = 2*pi*rand;
r = sqrt(rand);
y = (rc*r) * [cos(a) sin(a)] + x;
end