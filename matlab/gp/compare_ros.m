clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 3;
mode = 5;
w = 100;
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, '25');

%%

obj_pos = [517, 111];
base_pos = [397, 508];
base_theta = -0.002561858370170933;
load = [95.0, -20.0];
% cur = [513.0, 115.0];
% S = [539.09082792 116.051571;
%     539.07755951 116.04266902;
%     537.96407728 116.04060756];

figure(1)
clf
hold on
plot(obj_pos(1), obj_pos(2),'or');
% plot(cur(1), cur(2),'sm');

s = obj_pos - base_pos;
R = [cos(base_theta) -sin(base_theta); sin(base_theta) cos(base_theta)];

s = (R*s')';

s = [s, load];

s = (s-I.xmin(I.state_inx)) ./ (I.xmax(I.state_inx)-I.xmin(I.state_inx));

A = [1 1; 0 1; 1 0; 0 0];

for i = 1:size(A,1)-1
    a = A(i,:);
    [s_next, ~] = prediction(kdtree, Xtraining, s, a, I, 1);
    
    [idx, d] = knnsearch(kdtree, [s a], 'K', 100);
    dnn = Xtraining(idx,:);
    
    s_next = s_next(I.state_inx);
    s_next = (s_next .* (I.xmax(I.state_inx)-I.xmin(I.state_inx))) + I.xmin(I.state_inx);
    s_next = s_next(1:2);
    s_next = (R' * s_next')';
    s_next = s_next + base_pos(1:2);
    
    disp([a s_next])% norm(s_next-cur)])
    plot(s_next(1), s_next(2),'xb');%,S(i,1),S(i,2),'xg');
end

axis equal
hold off

%%

s = obj_pos - base_pos;
R = [cos(base_theta) -sin(base_theta); sin(base_theta) cos(base_theta)];
s = (R*s')';
s = [s, load];
s_start = (s-I.xmin(I.state_inx)) ./ (I.xmax(I.state_inx)-I.xmin(I.state_inx));

figure(2)
clf
hold on
plot(obj_pos(1), obj_pos(2),'or');

s = s_start;
S(1,:) = obj_pos;
for i = 1:3
    a = [1 1];
    [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);

    si = s;
    si = si(I.state_inx);
    si = (si .* (I.xmax(I.state_inx)-I.xmin(I.state_inx))) + I.xmin(I.state_inx);
    si = si(1:2);
    si = (R' * si')';
    si = si + base_pos(1:2);
    
    S(i+1,:) = si;
    
    plot(S(1:i,1),S(1:i,2),'.-g');
    drawnow;
end

s = s_start;
S(1,:) = obj_pos;
for i = 1:3
    a = [0 1];
    [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);

    si = s;
    si = si(I.state_inx);
    si = (si .* (I.xmax(I.state_inx)-I.xmin(I.state_inx))) + I.xmin(I.state_inx);
    si = si(1:2);
    si = (R' * si')';
    si = si + base_pos(1:2);
    
    S(i+1,:) = si;
    
    plot(S(1:i,1),S(1:i,2),'.-b');
    drawnow;
end

s = s_start;
S(1,:) = obj_pos;
for i = 1:3
    a = [1 0];
    [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);

    si = s;
    si = si(I.state_inx);
    si = (si .* (I.xmax(I.state_inx)-I.xmin(I.state_inx))) + I.xmin(I.state_inx);
    si = si(1:2);
    si = (R' * si')';
    si = si + base_pos(1:2);
    
    S(i+1,:) = si;
    
    plot(S(1:i,1),S(1:i,2),'.-k');
    drawnow;
end

hold off
axis equal