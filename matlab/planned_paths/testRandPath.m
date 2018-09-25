clear all
addpath('../../data/');

num = 14;
p = 1;0.26;

num = 1;
f = 'ab';

M1 = dlmread(['stateMeanPath_' num2str(num) '.txt'])*p;
S1 = dlmread(['stateStdPath_' num2str(num) '.txt'])*p;


figure(1)
clf
hold on
plot(M1(:,1),M1(:,2),'--b');
plot(M1(1,1),M1(1,2),'ob');


for i = 1:length(f)
    M2 = dlmread(['testedPath_' num2str(num) f(i) '.txt']);
    data2 = clear_stagnation(process_data(M2));
       
    plot(data2.obj_pos(:,1)*p,data2.obj_pos(:,2)*p,'.-r');
    plot(data2.obj_pos(1,1)*p,data2.obj_pos(1,2)*p,'or');
    
end
hold off
axis equal

%%
% S1 = S1(:,1:2);
% for i = 2:size(S1,1)
%     % http://www.ams.sunysb.edu/~zhu/ams570/Bayesian_Normal.pdf
%     S1(i,:) = S1(i-1,:).*S1(i,:);%sqrt((S1(i-1,:).^2.*S1(i,:).^2)./(S1(i-1,:).^2+S1(i,:).^2));
% end
    

figure(2)
subplot(211)
% errorbar(1:size(M1,1)-1,M1(1:end-1,1),S1(:,1),'-');
errorbar(1:size(M1,1)-1,M1(1:end-1,3),S1(:,3),'-');
hold on
for i = 1:length(f)
    M2 = dlmread(['testedPath_' num2str(num) f(i) '.txt']);
    data2 = clear_stagnation(process_data(M2));
%     plot(1:data2.n,data2.obj_pos(:,1)*p,'.-r');
    plot(1:data2.n,data2.act_load(:,1)*p,'.-r');
end
hold off
xlabel('Step');
% ylabel('x coordinate');
ylabel('Load act. 1');
legend('planned','actual path','location','northwest');
xlim([0 850]);
title(['Test path ' num2str(num)]);

subplot(212)
% errorbar(1:size(M1,1)-1,M1(1:end-1,2),S1(:,2),'-');
errorbar(1:size(M1,1)-1,M1(1:end-1,4),S1(:,4),'-');
hold on
for i = 1:length(f)
    M2 = dlmread(['testedPath_' num2str(num) f(i) '.txt']);
    data2 = clear_stagnation(process_data(M2));
%     plot(1:data2.n,data2.obj_pos(:,2)*p,'.-r');
    plot(1:data2.n,data2.act_load(:,2)*p,'.-r');

end
hold off
xlabel('Step');
% ylabel('y coordinate');
ylabel('Load act. 2');
legend('planned','actual path','location','northwest');
xlim([0 850]);


%%

function data = clear_stagnation(data)

M = data.obj_pos;

j = 1;
while norm(M(j,1:2)-M(j+1,1:2)) < 1e-1
    j = j + 1;
end

M(1:j,:) = [];
data.act_load(1:j,:) = [];

data.obj_pos = M;
data.n = size(M,1);

end
