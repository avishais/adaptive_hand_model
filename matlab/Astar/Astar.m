function path = Astar(start, goal)

properties;

% Load GD data
[Xtraining, ~, kdtree, I] = load_data(Mode, w, 1, '20');

do_plot = true;

path = [];

tree(1e6).state = [-1 -1];
tree(1).state = start;
tree(1).gCost = 0;
tree(1).fCost = heuristicCost(start, goal);
tree(1).parent = 0;
tree(1).action = [-100 -100];
tree_index = 1;

closedSet = [];
openSet = [1 tree(1).fCost tree(1).state];

gCost = [];

iter = 0;
while ~isempty(openSet)
    iter = iter + 1;

    [current, indexInOpenSet] = get_best_from_openset(openSet);
    
    if heuristicCost(tree(current).state, goal) < 1
        path = reconstruct_path(tree, current);
        plotTree(start, goal, tree, tree_index, path)
        return;
    end
    
    openSet(indexInOpenSet,:) = [];
    closedSet = [closedSet; current tree(current).state];
    
    neighbors = prop2neighbors(tree(current).state, kdtree, Xtraining, I); 
    
    for i = 1:size(neighbors,1)
        if is_close(neighbors(i,:), closedSet(:,(1:I.state_dim)+1)) %|| norm(neighbors(i,:) - [20 -15]) < 12
            continue;
        end
        
        tentative_gCost = tree(current).gCost + norm(neighbors(i,:)-tree(current).state);
        
        [b, openSet_inx] = is_close(neighbors(i,:), openSet(:,(1:I.state_dim)+2));
        if ~b
            tree_index = tree_index + 1;
            tree(tree_index).state = neighbors(i,:);
            tree(tree_index).parent =current;
            tree(tree_index).gCost = tentative_gCost;
            tree(tree_index).fCost = tentative_gCost + heuristicCost(neighbors(i,:), goal);
            openSet = [openSet; tree_index tree(tree_index).fCost tree(tree_index).state];
            neighbor_index = tree_index;
            disp(['Added node ' num2str(tree_index)]);
        else
            neighbor_index = openSet(openSet_inx, 1);
            
            neighbor_cost = openSet(openSet_inx,2);
            if tentative_gCost > neighbor_cost
                continue;
            end
        end
        
        tree(neighbor_index).parent = current;
        tree(neighbor_index).gCost = tentative_gCost;
        tree(neighbor_index).fCost = tentative_gCost + heuristicCost(neighbors(i,:), goal);
        
        if do_plot && ~mod(iter, 100)
            plotTree(start, goal, tree, tree_index)
            drawnow;
        end
    end
    
    
end
end

function S_next = prop2neighbors(s, kdtree, Xtraining, I)

properties;

S_next = zeros(size(A,1),length(s));
for i = 1:size(A,1)
    sn = normalize(s, I);
    [s_next, sigma] = prediction(kdtree, Xtraining, sn, A(i,:), I, 1);
    
    if ~check_bounds(s_next)
        continue;
    end
    
    S_next(i,:) = denormalize(s_next, I);    
end

S_next(all(S_next==0, 2),:) = [];

end

function b =check_bounds(s)
if any(s>1) || any(s<0)
    b = 0;
else
    b = 1;
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

function cost = heuristicCost(state, goal)

cost = norm((state - goal));

end

function [idx, indexInOpenSet] = get_best_from_openset(openSet)

[~, indexInOpenSet] = min(openSet(:,2));

idx = openSet(indexInOpenSet,1);

end

function path = reconstruct_path(tree, current)

path = [];

while current
    path = [tree(current).state; path];
    
    current = tree(current).parent;
end

end

function [b, inx] = is_close(state, set)

if isempty(set)
    b = 0;
    inx = -1;
    return;
end

d = (repmat(state, size(set, 1), 1) - set).^2;
d = sqrt(sum(d')');

[~,inx] = min(d);
if any(d < 0.08)
    b = 1;
else
    b = 0;
end

end


function plotTree(start, goal, tree, tree_size, path)

if nargin == 4
    path = [];
end

figure(1)
clf
hold on
plot(start(:,1),start(:,2),'sr');
plot(goal(:,1),goal(:,2),'sr');

for i = 2:tree_size
    j = tree(i).parent;
    plot([tree(i).state(1) tree(j).state(1)],[tree(i).state(2) tree(j).state(2)],'k');    
end

if ~isempty(path)
    plot(path(:,1),path(:,2),'-m');
end

hold off
axis equal
% axis([25 125 -420 -380]);
axis([60 90 -420 -390]);
grid on

end