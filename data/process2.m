clear all
files = dir(fullfile('./ca/', '*.txt'));
files = struct2cell(files)';
files = sortrows(files, 1);

% files(32:50,:) = [];

n = max(size(files));

% mode:
% 1 - state is only object position
% 2 - state is object and gripper tips positions
% 3 - state is object and all gripper positions
% 4 - state is object and actuator positions
% 5 - state is object position and actuator load
% 6 - state is object, gripper and actuator positions
% 7 - state is object, gripper and actuator positions, and load
% 8 - state is object and actuator positions and actuator load

%%
% mode = 1;
for mode = 1:8
    disp(['Processing data for feature conf. ' num2str(mode) '...']);
    Q = cell(n,1);
    P = [];
    DT = [];
    for i = 1:n
        f = files{i,1};
               
        D = dlmread(['./ca/' f], ' ');
        
        if strcmp(f, 'ca_25_test3.txt') % test path
            f;
        end
        
        % Clean data
        data = process_data(D);
        
        DT = [DT; data.dt];
        
        M = [];
        for j = 1:data.n-1
            
            % Check if there is contact/load and action
            if any(data.act_load(j,:)==0) || any(data.ref_vel(j,:)==0) %|| any(abs(D(j,6:7))-0.06 > 1e-2)
                if ~strcmp(f, 'ca_25_test2.txt') && ~strcmp(f, 'ca_25_test3.txt')
                    continue;
                end
            end
            
            % Check if there is change, if not, move on, or check if transition is corrupt seen as jump in state
            if norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) < 1e-4 || norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) > 30
                if ~strcmp(f, 'ca_25_test2.txt') && ~strcmp(f, 'ca_25_test3.txt')
                    continue;
                end
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
            end
            
        end
        
        Q{i}.data = M;
        Q{i}.dt = mean(DT);
        Q{i}.file = f;
        switch mode
            case 1
                Q{i}.action_inx = 3:4;
                Q{i}.state_inx = 1:2;
                Q{i}.state_nxt_inx = 5:6;
            case 2
                Q{i}.action_inx = 7:8;
                Q{i}.state_inx = 1:6;
                Q{i}.state_nxt_inx = 9:14;
            case 3
                Q{i}.action_inx = 11:12;
                Q{i}.state_inx = 1:10;
                Q{i}.state_nxt_inx = 13:22;
            case 4
                Q{i}.action_inx = 5:6;
                Q{i}.state_inx = 1:4;
                Q{i}.state_nxt_inx = 7:10;
            case 5
                Q{i}.action_inx = 5:6;
                Q{i}.state_inx = 1:4;
                Q{i}.state_nxt_inx = 7:10;
            case 6
                Q{i}.action_inx = 13:14;
                Q{i}.state_inx = 1:12;
                Q{i}.state_nxt_inx = 15:26;
            case 7
                Q{i}.action_inx = 15:16;
                Q{i}.state_inx = 1:14;
                Q{i}.state_nxt_inx = 17:30;
            case 8
                Q{i}.action_inx = 7:8;
                Q{i}.state_inx = 1:6;
                Q{i}.state_nxt_inx = 9:14;
        end
        %         if strcmp(f, 'c_25_7.txt') % test path
        %             i_s1 = size(P,1)+1 + 340;
        %             i_e1 = size(P,1)+size(M,1) - 350;
        %         end
        %         if strcmp(f, 'c_25_69.txt') % Second test path
        %             i_s2 = size(P,1)+1;
        %             i_e2 = size(P,1)+size(M,1);
        %         end
        if strcmp(f, 'ca_25_test.txt') % test path
            Xtest1.data = M;
            Xtest1.base_pos = data.base_pos;
            Xtest1.theta = data.theta;
        else
            if strcmp(f, 'ca_25_test2.txt') % test path
                Xtest2.data = M;
                Xtest2.base_pos = data.base_pos;
                Xtest2.theta = data.theta;            
            else
                if strcmp(f, 'ca_25_test3.txt') % test path
                    Xtest3.data = M;
                    Xtest3.base_pos = data.base_pos;
                    Xtest3.theta = data.theta;                
                else
                    P = [P; M];
                end
                
            end
        end
    end
       
    %%
    %     Xtest = P(i_s1:i_e1,:);
    %     Xtest2 = P(i_s2:i_e2,:);
    %     P(i_s1:i_e1,:) = [];
    Xtraining = P;
    
    
    %     dlmwrite(['Ca_25_train_' num2str(mode) '.db'], P, ' ');
    %     dlmwrite(['Ca_25_test_' num2str(mode) '.db'], Xtest, ' ');
    %     dlmwrite(['Ca_25_test2_' num2str(mode) '.db'], Xtest2, ' ');
    
    save(['Ca_25_' num2str(mode) '.mat'], 'Q', 'Xtraining', 'Xtest1', 'Xtest2','Xtest3');
end
%%
plot(P(:,1),P(:,2),'.k');
hold on
plot(Xtest1.data(:,1),Xtest1.data(:,2),'-r','linewidth',4);
plot(Xtest2.data(:,1),Xtest2.data(:,2),'-g','linewidth',4);
plot(Xtest3.data(:,1),Xtest3.data(:,2),'-b','linewidth',4);
hold off
legend('training','test_1','test_2','test_3');


