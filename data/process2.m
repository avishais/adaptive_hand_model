clear all
files = dir(fullfile('./', '*.txt'));

n = max(size(files));

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
    for j = 1:data.n-1
        
        % Check if there is contact/load and action
        if any(data.act_load(j,:)==0) || any(data.ref_vel(j,:)==0) %|| any(abs(D(j,6:7))-0.06 > 1e-2)
            continue;
        end
        
        % Check if there is change, if not, move on, or check if transition is corrupt seen as jump in state 
        if all(data.obj_pos(j,1:2)==data.obj_pos(j+1,1:2)) || norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) > 30
            continue;
        end
               
        % Currently take state as object position (no angle)
        % M = [(state,action), (state')];
        M = [M; [data.obj_pos(j,1:2), data.ref_vel(j,:), data.obj_pos(j+1,1:2)]]; % data 1
        % M = [M; [D(j,24:25), D(j, [16 20 18 22]), D(j,6:7), D(j+1, 24:25), D(j+1, [16 20 18 22]), i]];  % data 2
        
    end
    
    Q{i}.data = M; 
    Q{i}.dt = mean(DT);
    Q{i}.file = f;
    Q{i}.action_inx = 3:4;
    Q{i}.state_inx = 1:2;
    Q{i}.state_nxt_inx = 5:6;
%     Q{i}.action_inx = 7:8;
%     Q{i}.state_inx = 1:6;
%     Q{i}.state_nxt_inx = 9:14;
    P = [P; M];
end

%%
save('data_25.mat', 'Q', 'P');
dlmwrite('data_25.db', P, ' ');

%%
plot(P(:,[1:2]));