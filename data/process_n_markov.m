clear all
files = dir(fullfile('./', '*.txt'));

n = max(size(files));

% mode: 
% 2 - state is only object position
% 4 - state is object and gripper tips positions
% 6 - state is object and all gripper positions
mode = 2;
windowSize = 10;

Q = cell(n,1);
P = [];
DT = [];
for i = 1:n
    f = files(i).name;
    
    D = dlmread(f, ' ');
    
    % Clean data
    data = process_data(D);
    
    DT = [DT; data.dt];
    
    M = [];
    for j = windowSize+1:data.n-1
        
        % Check if there is contact/load and action
        if any(data.act_load(j,:)==0) || any(data.ref_vel(j,:)==0) %|| any(abs(D(j,6:7))-0.06 > 1e-2)
            continue;
        end
        
        % Check if there is change, if not, move on, or check if transition is corrupt seen as jump in state 
        if norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) < 1e-4 || norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) > 30
            continue;
        end
        
        % Just for checking
        if all(data.ref_vel(j,:)==0.06) && data.obj_pos(j,2) > data.obj_pos(j+1,2)
            continue;
        end
        
        K = [];
        for k = j-windowSize:j
            switch mode
                case 2
                    K = [K [data.obj_pos(k,1:2), data.ref_vel(k,:)]];
                case 4
                    K = [K [data.obj_pos(k,1:2), data.m1(k,:), data.m2(k,:), data.ref_vel(k,:)]];
                case 6
                    K = [K [data.obj_pos(k,1:2), data.m1(k,:), data.m2(k,:), data.m3(j,:), data.m4(j,:), data.ref_vel(k,:)]];
            end                    
        end
        % Currently take state as object position (no angle)
        % M = [(state,action), (state')];
        switch mode
            case 2
                M = [M; [K, data.obj_pos(j+1,1:2)]]; % data 1
            case 4
                M = [M; [K, data.obj_pos(j+1,1:2), data.m1(j+1,:), data.m2(j+1,:)]];  % data 2
            case 6
                M = [M; [K, data.obj_pos(j+1,1:2), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)]];  % data 3
        end

                        
    end
    
    Q{i}.data = M; 
    Q{i}.dt = mean(DT);
    Q{i}.file = f;
    switch mode
        case 2
            Q{i}.action_inx = 4*(windowSize+1)-1:4*(windowSize+1);
            Q{i}.state_inx = 1:4*(windowSize+1)-2;
            Q{i}.state_nxt_inx = 4*(windowSize+1)+1:4*(windowSize+1)+2;
        case 4
            Q{i}.action_inx = 8*windowSize-1:8*windowSize;
            Q{i}.state_inx = 1:8*windowSize-2;
            Q{i}.state_nxt_inx = 8*windowSize+1:8*windowSize+6;
        case 6
            Q{i}.action_inx = 12*windowSize-1:12*windowSize;
            Q{i}.state_inx = 1:12*windowSize-2;
            Q{i}.state_nxt_inx = 12*windowSize+1:12*windowSize+10;
    end
    P = [P; M];
end

%%
save(['data_25_' num2str(mode) '_w' num2str(windowSize) '.mat'], 'Q', 'P');
dlmwrite(['data_25_' num2str(mode) '_w' num2str(windowSize) '.db'], P, ' ');

%%
plot(P(:,[1:2]));