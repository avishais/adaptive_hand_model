clear all
files = dir(fullfile('./', '*.txt'));

n = max(size(files));

Q = cell(n,1);
P = [];
for i = 1:n
    f = files(i).name;
    
    D = dlmread(f, ',');
    
    % Clean 0's
    j = 1;
    while (j <= size(D,1))
        if (sum(D(j,2:end)==0) > 7)
            D(j,:) = [];
            continue;
        end
        j = j + 1;
    end
    
    M = [];
    for j = 1:size(D,1)-1
        
        % Check if there is contact/load
        if any(D(j,4:5)==0) || any(D(j,6:7)==0) || any(abs(D(j,6:7))-0.06 > 1e-2)
            continue;
        end
        
        % Check if there is change, if not, move on
        if all(D(j,24:25)==D(j+1,24:25))
            continue;
        end
        
        % Currently take state as object position
        % M = [(state,action), (state')];
        M = [M; [D(j,24:25), D(j, [16 20 18 22]), D(j,6:7), D(j+1, 24:25), D(j+1, [16 20 18 22]), i]];      
    end
    
    Q{i}.data = M;
    Q{i}.file = f;
    Q{i}.action_inx = 7:8;
    Q{i}.state_inx = 1:6;
    Q{i}.state_nxt_inx = 9:14;
    P = [P; M];
end

%%
save('data2.mat', 'Q', 'P');
dlmwrite('data2.db', P(:,1:end-1), ' ');

%%
plot(P(:,3:4));