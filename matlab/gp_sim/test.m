clear all

% file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont_0.db';
% file1 = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont.db';
% is_start = 12100;%250;
% is_end = 12600;%750;
% D = [dlmread(file); dlmread(file1)];
% Dtest = D(is_start:is_end,:);

file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont.db';
D = dlmread(file);

% D = D(1:340,:);
%%
% n = 10000;
% inx = randperm(size(D,1));
% D = D(inx(1:n),:);

% Dtest = Dtest(91:91+50,:);

%%

X = D;%(D(:,1)<-40 & D(:,1)>-70 & D(:,2)<108 & D(:,2)>94,:);

figure(1)
clf
hold on
plot(X(1,1),X(1,2),'sk','markerfacecolor','c');
plot(X(:,1),X(:,2),'.k');
% for i = 1:size(X,1)
%     plot(X(i,[1 7]),X(i,[2 8]),'-m');
% end
% plot(Dtest(:,1),Dtest(:,2),'-b');
hold off
axis equal

%%

H = zeros(size(D,1),1);
for i = 1:size(D,1)
    H(i) = norm(D(i,1:2)-D(i,7:8));
end
h = hist(H, 20);

max(H)