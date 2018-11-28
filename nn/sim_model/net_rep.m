function [W, b, x_max, x_min, activation] = net_rep()


% R = load('net_transition.netxt');
R = load('/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/net_transition.netxt');


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
% i = 22273+1;%randi(size(D,1));
% x = D(i,1:4);
% % x = x.*(x_max(1:4)-x_min(1:4)) + x_min(1:4)
% y_real = D(i,5:6)-D(i,1:2)
% 
% dx = Net(x, W, b, x_max, x_min)
% % y = [0.63304348 0.60869565] + Net(x, W, b, x_max, x_min)
% x_next = x(1:2)+dx