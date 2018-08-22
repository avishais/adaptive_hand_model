
addpath('../gp');

% These are in coordinates relative to the base frame (marker 0)
start = [71 -410 70 -70];
goal = [80 -400];

[path, action_path] = RRT(start, goal, 1e5);

figure(2)
clf
hold on
plot(start(1),start(2),'pk','markerfacecolor','r');
plot(goal(1),goal(2),'pk','markerfacecolor','g');
plot(path(:,1),path(:,2),'-m');
axis equal
%     axis([s_min(1) s_max(1) s_min(2) s_max(2)]);
axis([25 125 -420 -380]);
hold off

properties;
save('path.mat');