disp('Loading data...');

if 0
    % file = '../../data/sim_transition_data_discrete.db';
    file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_discrete.db';
    % file = 'hand_dis.db';
    is_start = 1;
    is_end = 260;%1900; %260;
    D = dlmread(file);  
    
%     n = 500000;    
%     Dtest = D(is_start:is_end,:);
%     D = D(is_end+1:end,:);
%     inx = randperm(size(D,1));
%     D = D(inx(1:n),:);
%     D = [Dtest; D];
    
    % D = [D(:,1:2) D(:,5:6) D(:,7:8)];
    
    save('data_discrete.mat', 'D', 'is_start', 'is_end');
else
    % file = '../../data/sim_transition_data_cont.db';
    file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont_0.db';
    file1 = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont.db';
    is_start = 12100;%250;
    is_end = 12600;%750;
    D = [dlmread(file); dlmread(file1)];
    
    n = 2000000;
    Dtest = D(is_start:is_end,:);
    D = D(is_end+1:end,:);
    inx = randperm(size(D,1));
    D = D(inx(1:n),:);
    D = [Dtest; D];
    
    save('data_cont.mat', 'D', 'is_start', 'is_end');
end

disp('Data saved.');