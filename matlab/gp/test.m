clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 3;
mode = 5;
w = 1;
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num);

%%

obj_pos = [569 126];
base_pos = [396 512];
base_theta = 0.0124993490194;
load = [168.0, -28.0];
cur = [570. 126.];
S = [498.85152694  99.7773701;
    498.92590975  99.6589929;
    498.91640462  99.62592367];

% obj_pos = [496 102];
% base_pos = [396 512];
% base_theta = pi - 0.0124993490194;
% load = [70.0, -18.0];
% cur = [505. 102.];
% S = [511.63133567 106.27826545;
%     490.70160199  99.77253334;
%     490.12765309 100.74103301; 490.12765309 100.74103301];

figure(1)
clf
hold on
plot(obj_pos(1), obj_pos(2),'or');
plot(cur(1), cur(2),'sm');


s = obj_pos - base_pos;
R = [cos(base_theta) -sin(base_theta); sin(base_theta) cos(base_theta)];

s = (R*s')';

s = [s, load];

s = (s-I.xmin(I.state_inx)) ./ (I.xmax(I.state_inx)-I.xmin(I.state_inx));

A = [1 1; 0 1; 1 0; 0 0];

for i = 1:size(A,1)-1
    a = A(i,:);
    [s_next, ~] = prediction(kdtree, Xtraining, s, a, I, 1);
    
    s_next = s_next(I.state_inx);
    s_next = (s_next .* (I.xmax(I.state_inx)-I.xmin(I.state_inx))) + I.xmin(I.state_inx);
    s_next = s_next(1:2);
    s_next = (R' * s_next')';
    s_next = s_next + I.base_pos(1:2);
    
    disp([a s_next norm(s_next-cur)])
    plot(s_next(1), s_next(2),'xb',S(i,1),S(i,2),'xg');
end

axis equal
hold off
