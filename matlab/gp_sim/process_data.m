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
    file1 = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont_1.db';
    is_start = 12100;%250;
    is_end = 12600;%750;
    D = [dlmread(file); dlmread(file1)];
    
    D = clean(D);
    
%     n = 500000;
%     Dtest = D(is_start:is_end,:);
%     D = D(is_end+1:end,:);
%     inx = randperm(size(D,1));
%     D = D(inx(1:n),:);
%     D = [Dtest; D];
%     is_end = size(Dtest,1);
%     is_start = 1;
    
    save('data_cont.mat', 'D', 'is_start', 'is_end');
end

disp('Data saved.');

function Dnew = clean(D)
% Cleans the jumps at/before dropping
disp('Cleaning...');

i = 1; j = 1;
inx = zeros(size(D,1),1);
while i<= size(D,1)
    disp(i);
    if norm(D(i,1:2)-D(i,7:8)) < 2
        inx(j) = i;
        j = j + 1;
    end
%     if i > 1 && all(D(i,3:4) == 12) % a new episode starts with loads = 12
%         D(i-10:i-1,:) = [];
%         i = i - 8;
%         for j = i-1:-1:i-20
%             if norm(D(j,1:2)-D(j,7:8)) > 1
%                 D(j,:) = [];
%                 i = i - 1;
%             end
%         end
%         i = i + 1;
%     else
        i = i + 1;
%     end
end

inx = inx(1:j-1);
Dnew = D(inx,:);

end