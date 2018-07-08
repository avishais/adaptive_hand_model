function [W, b, x_max, x_min] = net_rep(num_net)

if nargin == 0
    num_net = 1;
end

R = load(['../models/net' num2str(num_net) '.netxt']);
n = R(1);
activation = R(2);
R(1:2) = [];

%%
% Weights
for i = 1:n
    r = R(1);
    w = R(2);
    R(1:2)=[];
    
    Wt = R(1:r*w);
    R(1:r*w) = [];
    
    W{i} = reshape(Wt, [w, r])';
end

% Biases
for i = 1:n
    w = R(1);
    R(1)=[];
    
    bt = R(1:w);
    R(1:w) = [];
    
    b{i} = bt;
end

l = length(R);
x_max = R(1:l/2); x_min = R(l/2+1:end);

%% Test net

% D = dlmread('../data/data1.db');
% Q = load('../data/data1.mat', 'Q');
% Q = Q.Q;
% 
% P = Q{9}.data;
% 
% 
% i = randi(size(D,1));
% x = [0.48069498 0.48878924 0.         0.        ];%D(i,1:4);
% y_real = D(i,5:6)

% y = Net(x, W, b, x_max, x_min)