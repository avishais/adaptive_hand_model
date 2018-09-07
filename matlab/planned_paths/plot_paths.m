clear all
addpath('../../data/');


M1 = dlmread('cc_plan_path.txt');
M2 = dlmread('cc_planned_path.txt');
M3 = dlmread('planning_traj.txt',',');


data1 = process_data(M1);
data2 = process_data(M2);


figure(1)
clf
hold on
plot(data1.obj_pos(:,1),data1.obj_pos(:,2),'--b');
plot(data1.obj_pos(1,1),data1.obj_pos(1,2),'ob');
plot(data2.obj_pos(:,1),data2.obj_pos(:,2),'-r');
plot(data2.obj_pos(1,1),data2.obj_pos(1,2),'or');
plot(M3(:,1),M3(:,2),'-m');
hold off
axis equal