clear all

px2mm = 0.2621;

%%
data_source = 20;

mode = 8;

file = ['../data/Ca_' num2str(data_source) '_' num2str(mode)];
D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest1','Xtest2', 'Xtest3');
Q = D.Q;

action_inx = Q{1}.action_inx;
state_inx = Q{1}.state_inx;
state_nxt_inx = Q{1}.state_nxt_inx;

P = D.Xtest2.data;
P = P(700:end,:);

[W, b, x_max, x_min, activation] = net_rep(mode, data_source);

%% Open loop

figure(2)
clf
hold on

mse = 0;
for i = 2:size(P,1)-1
    x = P(i,[state_inx action_inx]);
    x_real_next = P(i,state_nxt_inx);
    
    x_next = x(state_inx) + Net(x, W, b, x_max, x_min, activation);
    
    mse = mse + norm(x_real_next(1:2)*px2mm-x_next(1:2)*px2mm)^2;
    
    plot(x(1),x(2),'xk');
    plot(x_next(1),x_next(2),'om');
    plot(x_real_next(1), x_real_next(2),'ob');
    plot([x(1) x_next(1)],[x(2) x_next(2)],'-k');
    
end
hold off

mse = mse/(size(P,1)-2);
rmse = sqrt(mse)

%%

function sd = denormz(s, I)

xmin = I.xmin(1:length(s));
xmax = I.xmax(1:length(s));

sd = s .* (xmax-xmin) + xmin;

end

function sd = project2image(s, I)

s = denormz(s, I);

% R = [cos(I.theta) -sin(I.theta); sin(I.theta) cos(I.theta)];
% s = (R' * s')';
% 
% sd = s + I.base_pos(1:2);

end
