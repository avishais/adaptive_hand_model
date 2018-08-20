clear all

filename = 'ca_25_temp';

M = dlmread(['./ca/' filename '.txt'],' ');
% M = dlmread('./berk_data/c_l_n_1.txt',',');

save2newFile = false;

% The columns are in the following order:
% 
% 1: Time
% 
% 2: actuator pos 1
% 3: actuator pos 2
% 4: act load 1
% 5: act load 2
% 6: act vel ref 1
% 7: act vel ref 2
% 8: act pos ref 1
% 9: act pos ref 2
% 10: finger pos x 1
% 11: finger pos y 1
% 12: finger pos x 2
% 13: finger pos y 2
% 14: finger pos x 3
% 15: finger pos y 1
% 16: finger pos x 4
% 17: finger pos y 4
% 18: obj pos x
% 19: obj pos y
% 20: obj orientation
% 21: Base angle

%% Clean

% M = M(1:200,:);

data = process_data(M);

D = [];
for i = 2:data.n
    if all(data.ref_vel(i,:)==0) && ~all(data.ref_vel(i-1,:)==0)
        D = [D; i];
    end
end
E = [];
for i = 2:data.n
    if all(data.ref_vel(i-1,:)==0) && ~all(data.ref_vel(i,:)==0)
        E = [E; i];
    end
end
E = [E; data.n];

M1 = zeros(length(D),3);
M2 = zeros(length(D),3);
for i = 1:length(D)
    M1(i,1) = data.act_load(D(i),1);
    M1(i,2) = data.act_load(E(i),1);
    M1(i,3) = (M1(i,2)-M1(i,1))/M1(i,1)*100;
    
    M2(i,1) = data.act_load(D(i),2);
    M2(i,2) = data.act_load(E(i),2);
    M2(i,3) = (M2(i,2)-M2(i,1))/M2(i,1)*100;
end
    




%%

figure(1)
clf
subplot(211)
plot(data.T, abs(data.act_load(:,1:2)), '.-');
hold on
for i = 1:length(D)
    plot([1 1]*data.T(D(i)),ylim,':k');
end
for i = 1:length(E)
    plot([1 1]*data.T(E(i)),ylim,':m');
end
hold off
xlabel('Time (sec)');
ylabel('Position');
legend('x','y');
title('Object position');
grid
xlim([0 data.T(end)]);

subplot(212)
plot(data.T,data.ref_vel);
xlabel('Time (sec)');
ylabel('Actuattors - velocities');
legend('actuator 1','actuator 2');
ylim([-0.07 0.07]);
grid
title('Actuation inputs');
xlim([0 data.T(end)]);

