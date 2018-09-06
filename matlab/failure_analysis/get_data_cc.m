clear all

addpath('../../data/');

data_source = '20';
files = dir(fullfile('../../data/cc/', ['cc_' data_source '_*.txt']));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);
files_size = size(files,1);

n = size(files,1);

mode = 8;
%%
M_drop = [];
M_ol = [];
M_normal = [];
L_drop = [];
L_ol = [];
L_normal = [];

drop_inx = [4 8 9 10 11 13 14 15 16 17 24 27 28 29 31 32 33 35 36 37 38 39 40 41 44 46 50 52 53 55 56 57 58 59 60 61 62 63 66 67 68 71 72 73 74];
ol_inx = [5 25 30 34 42 43 45 47 48 49 54 64 65 69];

for i = 1:n
    
    f = files{i,1};
    
    if strcmp(f, 'cc_20_test1.txt')
        continue;
    end
    
    D = dlmread(['../../data/cc/' f], ' ');
    
    inx = str2num(f(7:8));
    
    % Clean data
    data = process_data(D);
    
    M = [];
    for j = 1:data.n-1
        
        % Check if there is contact/load and action
        if all(data.ref_vel(j,:)==0) %|| any(abs(D(j,6:7))-0.06 > 1e-2) any(data.act_load(j,:)==0) ||
            continue;
        end
        
        % Check if there is change, if not, move on, or check if transition is corrupt seen as jump in state
        if norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) > 30 % norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) < 1e-4 ||
            continue;
        end
        
        % Just for checking
        %             if all(data.ref_vel(j,:)==0.06) && data.obj_pos(j,2) > data.obj_pos(j+1,2)
        %                 if ~strcmp(f, 'ca_25_test2.txt') && ~strcmp(f, 'ca_25_test3.txt')
        %                     continue;
        %                 end
        %             end
        
        % Currently take state as object position (no angle)
        % M = [(state,action), (state')];
        switch mode
            case 1
                M = [M; [data.obj_pos(j,1:2), data.ref_vel(j,:), data.obj_pos(j+1,1:2)]]; % data 1
            case 2
                M = [M; [data.obj_pos(j,1:2), data.m1(j,:), data.m2(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.m1(j+1,:), data.m2(j+1,:)]];  % data 2
            case 3
                M = [M; [data.obj_pos(j,1:2), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)]];  % data 2
            case 4
                M = [M; [data.obj_pos(j,1:2), data.act_pos(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:)]];
            case 5
                M = [M; [data.obj_pos(j,1:2), data.act_load(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:)]];
            case 6
                M = [M; [data.obj_pos(j,1:2), data.act_pos(j,:), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:),  data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)]];
            case 7
                M = [M; [data.obj_pos(j,1:2), data.act_pos(j,:), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:), data.act_load(j,:),  data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:), data.act_load(j+1,:)]];
            case 8
                M = [M; [data.obj_pos(j,1:2), data.act_pos(j,:), data.act_load(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.act_load(j+1,:)]];
            case 9
                M = [M; [data.obj_pos(j,1:2), data.act_load(j,:), data.m1(j,:), data.m2(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:), data.m1(j+1,:), data.m2(j+1,:)]];
            case 10
                M = [M; [data.obj_pos(j,1:2), data.act_load(j,:), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)]];
        end
    end
    
    switch mode
        case 1
            action_inx = 3:4;
            state_inx = 1:2;
            state_nxt_inx = 5:6;
        case 2
            action_inx = 7:8;
            state_inx = 1:6;
            state_nxt_inx = 9:14;
        case 3
            action_inx = 11:12;
            state_inx = 1:10;
            state_nxt_inx = 13:22;
        case 4
            action_inx = 5:6;
            state_inx = 1:4;
            state_nxt_inx = 7:10;
        case 5
            action_inx = 5:6;
            state_inx = 1:4;
            state_nxt_inx = 7:10;
        case 6
            action_inx = 13:14;
            state_inx = 1:12;
            state_nxt_inx = 15:26;
        case 7
            action_inx = 15:16;
            state_inx = 1:14;
            state_nxt_inx = 17:30;
        case 8
            action_inx = 7:8;
            state_inx = 1:6;
            state_nxt_inx = 9:14;
        case 9
            action_inx = 9:10;
            state_inx = 1:8;
            state_nxt_inx = 11:18;
        case 10
            action_inx = 13:14;
            state_inx = 1:12;
            state_nxt_inx = 15:26;
    end
    
    if any(inx==drop_inx)
        M_drop = [M_drop; M(end, [state_inx action_inx])];
        L_drop = [L_drop; 0];
        
        u = floor(size(M,1)/2);
        M_normal = [M_normal; M(u, [state_inx action_inx])];
        L_normal = [L_normal; 1];
    end
    if any(inx==ol_inx)
        M_ol = [M_ol; M(end, [state_inx action_inx])];
        L_ol = [L_ol; 0];
        
        u = floor(size(M,1)/2);
        M_normal = [M_normal; M(u, [state_inx action_inx])];
        L_normal = [L_normal; 1];
    end
    
    Q{i} = M;
    
end

clear data
data = [M_drop; M_ol; M_normal];
labels = [L_drop; L_ol; L_normal];

save(['class_data_cc_ ' num2str(mode) '.mat'],'data','labels','Q');