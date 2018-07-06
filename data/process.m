% M = dlmread('r_l_n_2.txt',',');
% M = dlmread('c_l_n_1.txt',',');
M = dlmread('r_m_d_2.txt',',');


% The columns are in the following order:
% 
% 0: Time
% 
% 1: actuator pos 1
% 2: actuator pos 2
% 3: act load 1
% 4: act load 2
% 5: act vel ref 1
% 6: act vel ref 2
% 7: act pos ref 1f
% 8: act pos ref 2
% 9: cartesian vel ref 1
% 10: cartesian vel ref 2
% 11: finger angle 1
% 12: finger angle 2
% 13: finger angle 3
% 14: finger angle 4
% 15: finger pos x 1
% 16: finger pos x 2
% 17: finger pos x 3
% 18: finger pos x 4
% 19: finger pos y 1
% 20: finger pos y 2
% 21: finger pos y 3
% 22: finger pos y 4
% 23: obj pos x
% 24: obj pos y
% 25: obj orientation

%% Clean 0's
i = 1;
while (i <= size(M,1))
    if (sum(M(i,2:end)==0) > 7)
        M(i,:) = [];
        continue;
    end
    i = i + 1;
end

%% Clean steps
% i = 2;
% while (i <= size(M,1))
%     if ( all(M(i,24:25)==M(i-1,24:25)) )
%         M(i,:) = [];
%         continue;
%     end
%     i = i + 1;
% end

% M = M(1:21:end,:);

%% 
% T = M(:,1)*1e-6;
% T = T-T(1);
dt = 0.033;
T = (0:dt:dt*(size(M,1)-1))';

%%
figure(1)
subplot(211)
plot(T, M(:,24:25), '.-');
% plot(M(:,24), M(:,25));
xlabel('Time (sec)');
ylabel('Position');
legend('x','y');
title('Object position');
grid

subplot(212)
plot(T,M(:,[6 7]));
xlabel('Time (sec)');
ylabel('Actuattors - velocities');
legend('actuator 1','actuator 2');
ylim([-0.07 0.07]);
grid
title('Actuation inputs');

figure(2)
plot(T, M(:,4:5));
xlabel('Time (sec)');
ylabel('Load');
legend('actuator 1','actuator 2');