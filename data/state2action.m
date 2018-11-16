clear all

data_source = '20';
dataset = 'ce';

files = dir(fullfile(['./' dataset '/'], [dataset '_' data_source '_*.txt']));
files = struct2cell(files)';
files = sortrows(files, 1);

n = size(files,1);

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
for mode = 5%:8%[1 5 8]
    disp(['Processing data for feature conf. ' num2str(mode) '...']);
    Q = cell(n,1);
    P = [];
    DT = [];
    for i = 1:n
        f = files{i,1};
        
        D = dlmread(['./' dataset '/' f], ' ');
        
        % Clean data
        data = process_data(D);
        data.act_pos = MovingAvgFilter(data.act_pos, 30);
        
        figure(3)
        subplot(211)
        plot(data.act_pos(1:end,:));
        subplot(212)
        plot(diff(data.act_pos(1:end,:)));   
              
        DT = [DT; data.dt];
        
        M = [];
        for j = 2:data.n-1
            
            % Check if there is contact/load and action
            if all(data.ref_vel(j,:)==0) %|| any(abs(D(j,6:7))-0.06 > 1e-2) any(data.act_load(j,:)==0) || 
                if ~strcmp(f, 'ca_25_test2.txt') && ~strcmp(f, 'ca_25_test3.txt')
                    continue;
                end
            end
            
            % Check if there is change, if not, move on, or check if transition is corrupt seen as jump in state
            if norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) > 30 % norm(data.obj_pos(j,1:2)-data.obj_pos(j+1,1:2)) < 1e-4 ||
                if ~strcmp(f, 'ca_25_test2.txt') && ~strcmp(f, 'ca_25_test3.txt')
                    continue;
                end
            end
            
            
            % Currently take state as object position (no angle)
            % M = [(state,action), (state')];
            switch mode
                case 5
                    M = [M; [data.obj_pos(j,1:2), data.act_load(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_load(j+1,:)]];
                case 8
                    M = [M; [data.obj_pos(j,1:2), data.act_pos(j,:), data.act_load(j,:), data.ref_vel(j,:), data.obj_pos(j+1,1:2), data.act_pos(j+1,:), data.act_load(j+1,:)]];
            end
        end
        
        Q{i}.data = M;
        Q{i}.dt = mean(DT);
        Q{i}.file = f;
        switch mode
            case 5
                Q{i}.action_inx = 5:6;
                Q{i}.state_inx = 1:4;
                Q{i}.state_nxt_inx = 7:10;
            case 8
                Q{i}.action_inx = 7:8;
                Q{i}.state_inx = 1:6;
                Q{i}.state_nxt_inx = 9:14;
        end
        
        flag = 1;
        if strcmp(f, 'ce_20_test1.txt') % test path
            Xtest1.data = M;
            Xtest1.base_pos = data.base_pos;
            Xtest1.theta = data.theta;
            flag = 0;
        end
        if flag
            P = [P; M];
        end
        
    end
    
    %%
    Xtraining = P;
    
    %     dlmwrite(['Ca_25_train_' num2str(mode) '.db'], P, ' ');
    %     dlmwrite(['Ca_25_test_' num2str(mode) '.db'], Xtest, ' ');
    %     dlmwrite(['Ca_25_test2_' num2str(mode) '.db'], Xtest2, ' ');
    
%     save(['C' dataset(2) '_' data_source '_' num2str(mode) '.mat'], 'Q', 'Xtraining');
end
%%
plot(P(:,1),P(:,2),'.k');
legend('training','test_1','test_2','test_3');

%% Report

fprintf('---------------------------------------------------\n');
fprintf('Generated dataset with %d transition points.\n',size(Xtraining, 1));
A = unique(Xtraining(:,Q{1}.action_inx), 'rows');
A = sortrows(A,1);
H = zeros(size(A,1),1);
for i = 1:size(Xtraining,1)
    H = H + all(A==Xtraining(i,Q{1}.action_inx), 2);
end
Keys = {'k','W','D','k','k','A','X','k'};
for i = 1:length(H)
    fprintf([Keys{i} ': %d\n'], H(i));
end


bar(H);
set(gca,'xticklabel', Keys);

%%

function y = MovingAvgFilter(x, windowSize)

if nargin==1
    windowSize = 13;
end

y = x;

w = floor(windowSize/2);
for j = 1:size(x,2)
    for i = w+1:size(x,1)-w-1
        
        y(i,j) = sum(x(i-w:i+w,j))/length(i-w:i+w);
        
    end
    
end
end
