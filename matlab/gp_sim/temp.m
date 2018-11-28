file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont_0.db';
D = dlmread(file);

%%
i = 1;
while i<= size(D,1)
    if i > 1 && all(D(i,3:4) == 12)
        for j = i-1:-1:i-20
            if norm(D(j,1:2)-D(j,7:8)) > 1
                D(j,:) = [];
                i = i - 1;
            end
        end
        i = i + 1;
    else
        i = i + 1;
    end
end

%%

H = zeros(size(D,1),1);
for i = 1:size(D,1)
    H(i) = norm(D(i,1:2)-D(i,7:8));
end
hist(H,20);
h = hist(H, 20);

%%

% dlmwrite(file, D, ' ');