function [path, action_path] = RRT(start, goal, imax)

properties;

% Load GD data
[Xtraining, ~, kdtree, I] = load_data(Mode, w, 1, '25');

tree = start;
actions = [0, 0];
parent = 0;
i = 1;
tree_size = 1;
approxdist = norm(start(1:2)-goal);
approxsol = 1;

found_solution = false;

min_goal_distance = 8;

do_plot = false;
if do_plot
    figure(1)
    clf
    hold on
    plot(start(1),start(2),'pk','markerfacecolor','r');
    plot(goal(1),goal(2),'pk','markerfacecolor','g');
    axis equal
    %     axis([s_min(1) s_max(1) s_min(2) s_max(2)]);    
    axis([25 125 -420 -380]);
end

while i < imax
    disp(i);
    
    s_rand = rand_state();
    
    idx = knnsearch(tree, s_rand);
    
    s = tree(idx,:);
    
    % Choose random action
    a = random_action();
    
    % Propagate by GP prediction
    sn = normalize(s, I);
    [s_next, sigma] = prediction(kdtree, Xtraining, sn, a, I, 1);
    s_next = denormalize(s_next, I);
    
    % => check here for good state
    is_good_state = checkLimits(s_next);
    
    % If good state, add to tree
    if (is_good_state)
        tree_size = tree_size + 1;
        tree(tree_size,:) = s_next;
        actions(tree_size,:) = a;
        parent(tree_size) = idx;
        
        dist = norm(s_next(1:2)-goal);
        if dist < min_goal_distance
            disp('Found solution.');
            found_solution = true;
        else
            if dist < approxdist
                approxdist = dist;
                approxsol = tree_size;                
            end
        end  
        
        if do_plot && mod(i,20)==0
            plot([s(1) s_next(1)],[s(2) s_next(2)],'-r');
            drawnow;
        end
    end
    
    i = i + 1;
end

if do_plot
    hold off
end

if found_solution
    path = tree(end,:);
    action_path = actions(end,:);
    curnode = parent(end);
else
    path = tree(approxsol,:);
    action_path = actions(approxsol,:);
    curnode = parent(approxsol);
end

while curnode~=0
    path = [tree(curnode,:); path];
    action_path = [actions(curnode,:); action_path];
    curnode = parent(curnode);
end

end

function rstate = rand_state() 

properties;

rstate = rand(1,4).*(s_max-s_min) + s_min;

end

function a = random_action()

properties;

n = size(A,1);
a = A(randi(n),:);

end

function b = checkLimits(state)

properties;

if all(state >= s_min) && all(state <= s_max)
    b = 1;
else
    b = 0;
end

end

function xn = normalize(x, I)

n = length(x); 
xn = (x-I.xmin(1:n))./(I.xmax(1:n)-I.xmin(1:n));

end

function x = denormalize(xn, I)

n = length(xn); 
x = xn.*(I.xmax(1:n)-I.xmin(1:n)) + I.xmin(1:n);

end
