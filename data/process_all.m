clear all
files = dir(fullfile('./ca/', '*.txt'));
files = struct2cell(files)';
files = sortrows(files, 1);

n = size(files,1);

% mode:
% 1 - state is only object position
% 2 - state is object and gripper tips positions (markers 1 & 2)
% 3 - state is object and all gripper positions (markers 1, 2, 3, 4)
% 4 - state is object and actuator positions
% 5 - state is object position and actuator load
% 6 - state is object, gripper and actuator positions
% 7 - state is object, gripper and actuator positions, and load
% 8 - state is object and actuator positions and actuator load
% 9 - state is object position, gripper load and gripper tips position (markers 1 & 2)
% 10 - state is object position, gripper load and all gripper positions (markers 1, 2, 3, 4)

%%
% mode = 1;
vec_size = [6, 14, 22, 10, 10, 26, 30, 14, 22, 26, 16]+1;

for mode = 7%[5 8 9 10 11]
    disp(['Processing data for feature conf. ' num2str(mode) '...']);
    Q = cell(n,1);
    P = [];
    DT = [];
    for i = 1:n
        f = files{i,1};
        
        cyl_diameter = str2num(f(4:5));
        
%         if strcmp(f, 'ca_25_test2.txt') && strcmp(f, 'ca_25_test3.txt') && strcmp(f, 'ca_35_test1.txt')
%             continue;
%         end
        
        D = dlmread(['./ca/' f], ' ');
        
        % Clean data
        data = process_data(D);
        
        DT = [DT; data.dt];
        
        M = zeros(data.n-1, vec_size(mode));
        j = 1; k = 1;
        for j = 1:data.n-1
            
            % Check if there is contact/load and action
            if any(data.ref_vel(j,:)==0) %|| any(abs(D(j,6:7))-0.06 > 1e-2) any(data.act_load(j,:)==0) || 
%                 if ~strcmp(f, 'ca_15_test1.txt') && ~strcmp(f, 'ca_15_test2.txt') && ~strcmp(f, 'ca_15_test3.txt') && ~strcmp(f, 'ca_30_test1.txt') && ~strcmp(f, 'ca_30_test2.txt') && ~strcmp(f, 'ca_srp_test1.txt')
                    continue;
%                 end
            end
            
            % Check if there is change, if not, move on, or check if transition is corrupt seen as jump in state
            if norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) > 30 % norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) < 1e-4 ||
%                 if ~strcmp(f, 'ca_15_test1.txt') && ~strcmp(f, 'ca_15_test2.txt') && ~strcmp(f, 'ca_15_test3.txt') && ~strcmp(f, 'ca_30_test1.txt') && ~strcmp(f, 'ca_30_test2.txt') && ~strcmp(f, 'ca_srp_test1.txt')
                    continue;
%                 end
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
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.ref_vel(j,:), data.obj_pos(j+1,1:2)]; % data 1
                case 2
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.m1(j,:), data.m2(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.m1(j+1,:), data.m2(j+1,:)];  % data 2
                case 3
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)];  % data 2
                case 4
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_pos(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:)];
                case 5
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_load(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:)];
                case 6
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_pos(j,:), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:),  data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)];
                case 7
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_pos(j,:), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:), data.act_load(j,:),  data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:), data.act_load(j+1,:)];
                case 8
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_pos(j,:), data.act_load(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.act_load(j+1,:)];
                case 9
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_pos(j,:), data.act_load(j,:), data.m1(j,:), data.m2(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.act_load(j+1,:), data.m1(j+1,:), data.m2(j+1,:)];
                case 10
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_load(j,:), data.m1(j,:), data.m2(j,:), data.m3(j,:), data.m4(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:), data.m1(j+1,:), data.m2(j+1,:), data.m3(j+1,:), data.m4(j+1,:)];
                case 11
                    M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_pos(j,:), data.act_load(j,:), norm(data.m1(j,:)-data.m2(j,:)), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.act_load(j+1,:), norm(data.m1(j+1,:)-data.m2(j+1,:))];
%                     M(k,:) = [cyl_diameter, data.obj_pos(j,1:2), data.act_load(j,:), norm(data.m1(j,:)-data.m2(j,:)), norm(data.m3(j,:)-data.m4(j,:)), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:), norm(data.m1(j+1,:)-data.m2(j+1,:)), norm(data.m3(j+1,:)-data.m4(j+1,:))];
%                     M(k,:) = [cyl_diameter data.obj_pos(j,1:2), data.act_pos(j,:), data.act_load(j,:), data.ref_vel(j,:), cyl_diameter, data.obj_pos(j+1,1:2), data.act_pos(j,:), data.act_load(j+1,:)];         
            end
            
            k = k + 1;
            
        end
        M = M(1:k-1,:);
        
        Q{i}.data = M(:,2:end);
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
            case 9
                Q{i}.action_inx = 11:12;
                Q{i}.state_inx = 1:10;
                Q{i}.state_nxt_inx = 13:22;
            case 10
                Q{i}.action_inx = 13:14;
                Q{i}.state_inx = 1:12;
                Q{i}.state_nxt_inx = 15:26;
            case 11
                Q{i}.action_inx = 8:9;
                Q{i}.state_inx = 1:7;
                Q{i}.state_nxt_inx = 10:16;
%                 Q{i}.action_inx = 7:8;
%                 Q{i}.state_inx = 1:6;
%                 Q{i}.state_nxt_inx = 9:14;
%                 Q{i}.action_inx = 7:9;
%                 Q{i}.state_inx = 1:6;
%                 Q{i}.state_nxt_inx = 10:15;
        end
        
        flag = 1;
        if strcmp(f, 'ca_15_test1.txt') % test path
            Xtest_15_1.data = M(:,2:end);
            Xtest_15_1.base_pos = data.base_pos;
            Xtest_15_1.theta = data.theta;
            Xtest_15_1.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_15_test2.txt') % test path
            Xtest_15_2.data = M(:,2:end);
            Xtest_15_2.base_pos = data.base_pos;
            Xtest_15_2.theta = data.theta;
            Xtest_15_2.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_15_test3.txt') % test path
            Xtest_15_3.data = M(:,2:end);
            Xtest_15_3.base_pos = data.base_pos;
            Xtest_15_3.theta = data.theta;
            Xtest_15_3.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_30_test1.txt') % test path
            Xtest_30_1.data = M(:,2:end);
            Xtest_30_1.base_pos = data.base_pos;
            Xtest_30_1.theta = data.theta;
            Xtest_30_1.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_30_test2.txt') % test path
            Xtest_30_2.data = M(:,2:end);
            Xtest_30_2.base_pos = data.base_pos;
            Xtest_30_2.theta = data.theta;
            Xtest_30_2.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_11_test1.txt') % test path - small sharpie
            Xtest_11_1.data = M(:,2:end);
            Xtest_11_1.base_pos = data.base_pos;
            Xtest_11_1.theta = data.theta;
            Xtest_11_1.file = f;
            flag = 0;
        end
         if strcmp(f, 'ca_31_test1.txt') % test path - acrylic paint
            Xtest_31_1.data = M(:,2:end);
            Xtest_31_1.base_pos = data.base_pos;
            Xtest_31_1.theta = data.theta;
            Xtest_31_1.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_31_test2.txt') % test path - acrylic paint
            Xtest_31_2.data = M(:,2:end);
            Xtest_31_2.base_pos = data.base_pos;
            Xtest_31_2.theta = data.theta;
            Xtest_31_2.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_30_test3.txt') % test path - Glue stick
            Xtest_30_3.data = M(:,2:end);
            Xtest_30_3.base_pos = data.base_pos;
            Xtest_30_3.theta = data.theta;
            Xtest_30_3.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_15_test4.txt') % test path - thick sharpy
            Xtest_15_4.data = M(:,2:end);
            Xtest_15_4.base_pos = data.base_pos;
            Xtest_15_4.theta = data.theta;
            Xtest_15_4.file = f;
            flag = 0;
        end
        if strcmp(f, 'ca_45_test1.txt') % test path - thick sharpy
            Xtest_45_1.data = M(:,2:end);
            Xtest_45_1.base_pos = data.base_pos;
            Xtest_45_1.theta = data.theta;
            Xtest_45_1.file = f;
            flag = 0;
        end
        if flag
            P = [P; M];
        end
        
    end
    
    %%
    if 0
        C = unique(P(:,1));
        h = hist(P(:,1));
        h(h==0) = [];
        N = min(h);
        P_new = [];
        for i = 1:length(C)
            G = P(P(:,1)==C(i),2:end);
            p = randperm(size(G,1));
            p = p(1:min(N, size(G,1)));
            G = G(p,:);
            P_new = [P_new; G];
        end
        P = P_new;
    else
        P = P(:,2:end);
    end
    
    %%
    Xtraining = P;
    
    save(['Ca_all_' num2str(mode) '.mat'], 'Q', 'Xtraining', 'Xtest_15_1', 'Xtest_15_2', 'Xtest_15_3','Xtest_30_1','Xtest_30_2', 'Xtest_11_1','Xtest_31_1','Xtest_31_2','Xtest_30_3','Xtest_15_4','Xtest_45_1');
end
%%
plot(P(:,1),P(:,2),'.k');
hold on
plot(Xtest_15_1.data(:,1),Xtest_15_1.data(:,2),'-r','linewidth',4);
plot(Xtest_15_2.data(:,1),Xtest_15_2.data(:,2),'-g','linewidth',4);
plot(Xtest_15_3.data(:,1),Xtest_15_3.data(:,2),'-c','linewidth',4);
plot(Xtest_30_1.data(:,1),Xtest_30_1.data(:,2),'-b','linewidth',4);
plot(Xtest_30_2.data(:,1),Xtest_30_2.data(:,2),'-y','linewidth',4);
plot(Xtest_11_1.data(:,1),Xtest_11_1.data(:,2),'-y','linewidth',4);
plot(Xtest_31_1.data(:,1),Xtest_31_1.data(:,2),'-b','linewidth',4);
plot(Xtest_31_2.data(:,1),Xtest_31_2.data(:,2),'-y','linewidth',4);
plot(Xtest_30_3.data(:,1),Xtest_30_3.data(:,2),'-y','linewidth',4);
plot(Xtest_15_4.data(:,1),Xtest_15_4.data(:,2),'-y','linewidth',4);
plot(Xtest_45_1.data(:,1),Xtest_45_1.data(:,2),'-y','linewidth',4);
hold off
legend('training','Xtest_15_1','Xtest_15_2','Xtest_15_3','Xtest_30_1','Xtest_30_2','Xtest_11_1','Xtest_31_1','Xtest_31_2','Xtest_30_3','Xtest_15_4','Xtest_45_1');


