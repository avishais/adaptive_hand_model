clear all
addpath('../../data/');

num = 19;
p = 1;0.26;

M1 = dlmread(['traj' num2str(num) '.txt_edited'])*p;
S1 = dlmread(['std' num2str(num) '.txt'])*p;


figure(1)
clf
hold on
plot(M1(:,1),M1(:,2),'--b');
plot(M1(1,1),M1(1,2),'ob');


f = 'a';
for i = 1:length(f)
    
    M2 = dlmread(['planned_' num2str(num) f(i) '.txt']);
    data2 = process_data(M2);
    plot(data2.obj_pos(:,1)*p,data2.obj_pos(:,2)*p,'.-r');
    plot(data2.obj_pos(1,1)*p,data2.obj_pos(1,2)*p,'or');
    
end
hold off
axis equal

%%
% S1 = S1(:,1:2);

for i = 2:size(S1,1)
    S1(i,:) = sqrt((S1(i-1,:).^2.*S1(i,:).^2)./(S1(i-1,:).^2+S1(i,:).^2));
end
figure(3)
plot(S1);
    

figure(2)
subplot(211)
errorbar(1:size(M1,1)-1,M1(1:end-1,1),S1(:,1),'-');
hold on
for i = 1:length(f)
    
    M2 = dlmread(['planned_' num2str(num) f(i) '.txt']);
    data2 = process_data(M2);
    plot(1:data2.n,data2.obj_pos(:,1)*p,'.-r');
    
end
hold off
xlabel('Step');
ylabel('x coordinate');
legend('planned','actual path','location','northwest');

subplot(212)
errorbar(1:size(M1,1)-1,M1(1:end-1,2),S1(:,2),'-');
hold on
for i = 1:length(f)
    
    M2 = dlmread(['planned_' num2str(num) f(i) '.txt']);
    data2 = process_data(M2);
    plot(1:data2.n,data2.obj_pos(:,2)*p,'.-r');
    
end
hold off
xlabel('Step');
ylabel('y coordinate');
legend('planned','actual path','location','northwest');
