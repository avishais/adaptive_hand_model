clear all

addpath('/home/avishai/Documents/workspace/transition_model/data/');
%%

Qplanned = dlmread('traj1.txt',',');

rawData = dlmread('appliedTraj1.txt');
data = process_data(rawData);

Qreal = data.obj_pos(:,1:2);

%%

figure(1)
clf
hold on
plot(Qplanned(:,1),Qplanned(:,2),'.-r');
plot(Qreal(:,1),Qreal(:,2),'.-b');
hold off
