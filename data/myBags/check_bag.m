clear all

bag = rosbag('c_2018-07-20-17-56-00.bag');

% selectOptions = {'Time', [bag.StartTime bag.StartTime+30], 'Topic', {'/gripper/pos', '/gripper_t42/vel_ref_monitor', '/gripper_t42/pos_ref_monitor'}};
% bagselect = select(bag, selectOptions{:});

bagselect1 = select(bag, 'Topic', '/gripper_t42/vel_ref_monitor');
bagselect2 = select(bag, 'Topic', '/gripper_t42/pos_ref_monitor');
bagselect3 = select(bag, 'Topic', '/gripper/pos');
bagselect4 = select(bag, 'Topic', '/marker_tracker/image_space_pose_msg');

Tvel_ref = (table2array(bagselect1.MessageList(:,1)));
Tpos_ref = (table2array(bagselect2.MessageList(:,1)));
Tpos = (table2array(bagselect3.MessageList(:,1)));
Tm =  (table2array(bagselect4.MessageList(:,1)));


t_init = min([Tvel_ref(1), Tpos_ref(1), Tpos(1), Tm(1)]);
Tvel_ref = Tvel_ref - t_init;
Tpos_ref = Tpos_ref - t_init;
Tpos = Tpos - t_init;
Tm = Tm - t_init;

%%
msgs = readMessages(bagselect1);
vel_ref(:,1) = cellfun(@(m) double(m.Data(1)), msgs);
vel_ref(:,2) = cellfun(@(m) double(m.Data(2)), msgs);


figure(1)
title('Reference velocity')
plot(Tvel_ref, vel_ref);

%%
msgs = readMessages(bagselect2);
for i = 1:length(Tpos_ref)
    if isempty(msgs{i}.Data)
        pos_ref(i,:) = [0 0];
    else
        pos_ref(i,:) = msgs{i}.Data;
    end
end

msgs = readMessages(bagselect3);
for i = 1:length(Tpos)
    if isempty(msgs{i}.Data)
        pos(i,:) = [0 0];
    else
        pos(i,:) = msgs{i}.Data;
    end
end

figure(2)
subplot(1,2,1)
title('Position vs time')
plot(Tpos_ref, pos_ref,'--');
hold on
plot(Tpos, pos,'-');
hold off
subplot(1,2,2)
title('Position')
plot(pos_ref(:,1), pos_ref(:,2),'o--r');
hold on
plot(pos(:,1), pos(:,2),'o-b');
hold off
axis equal

%%
% msgs = readMessages(bagselect4);

%%



figure(3)
plot(Tvel_ref,'-b');
hold on
plot(Tpos_ref,'--r');
plot(Tpos,'.-g');
plot(Tm,'-m');
hold off

