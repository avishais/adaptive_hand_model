clear all

filename = 'ca_30_test2';

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

if save2newFile
    
    Q = [];
    for i = 1:data.n-1
        Q = [Q; data.obj_pos(i,1:2) data.act_load(i,1:2) data.ref_vel(i,:) data.obj_pos(i+1,1:2) data.act_load(i+1,1:2)];
    end
    
    dlmwrite([filename '_processed.txt'],Q , ' ');
    
end

%%
figure(1)
subplot(211)
plot(data.T, data.obj_pos(:,1:2), '.-');
xlabel('Time (sec)');
ylabel('Position');
legend('x','y');
title('Object position');
grid

subplot(212)
plot(data.T,data.ref_vel);
xlabel('Time (sec)');
ylabel('Actuattors - velocities');
legend('actuator 1','actuator 2');
ylim([-0.07 0.07]);
grid
title('Actuation inputs');

% figure(2)
% plot(data.T, data.act_load);
% xlabel('Time (sec)');
% ylabel('Load');
% legend('actuator 1','actuator 2');

% figure(4)
% clf
% hold on
% plot(data.T, data.obj_pos(:,1), '-c',data.T, data.obj_pos(:,2), '--c');
% plot(data.T, data.m1(:,1),'-r',data.T, data.m1(:,2),'--r');
% plot(data.T, data.m2(:,1),'-g',data.T, data.m2(:,2),'--g');
% plot(data.T, data.m3(:,1),'-b',data.T, data.m3(:,2),'--b');
% plot(data.T, data.m4(:,1),'-k',data.T, data.m4(:,2),'--k');
% hold off
% xlabel('Time (sec)');
% ylabel('Position');
% legend('obj - x','obj - y','marker 1 - x','marker 1 - y','marker 2 - x','marker 2 - y','marker 3 - x','marker 3 - y','marker 4 - x','marker 4 - y');

%%
% for i = data.n-60:2:data.n
for i = data.n%[1:10:data.n ]%1:50:
    i
    figure(3)
    clf
    circle(data.obj_pos(i,1),data.obj_pos(i,2),15,'r');
    circle(data.m1(i,1),data.m1(i,2),7,'g');
    circle(data.m2(i,1),data.m2(i,2),7,'b');
    circle(data.m3(i,1),data.m3(i,2),7,'k');
    circle(data.m4(i,1),data.m4(i,2),7,'m');
    circle(data.new_base_pos(1),data.new_base_pos(2),9,'c');
    hold on
    plot(data.obj_pos(1:i,1),data.obj_pos(1:i,2),'.-k');
    quiver(data.obj_pos(i,1),data.obj_pos(i,2),cos(data.obj_pos(i,3)),sin(data.obj_pos(i,3)),50,'k');
    quiver(data.new_base_pos(1),data.new_base_pos(2),cos(data.new_base_pos(3)),sin(data.new_base_pos(3)),50,'k');
    hold off
    text(150, -50, num2str(data.ref_vel(i,:)));
    axis equal
%     axis([-200 300 -450 10]);
    
    drawnow;
end


function circle(x,y,r,color)
    hold on
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    patch(xunit, yunit,color);
    hold off
end