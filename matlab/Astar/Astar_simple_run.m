start = [ 0 0 ];
goal = [30 -32];

path = Astar_simple(start, goal);


function path = Astar_simple(start, goal)

path = [];

A = [1 0; -1 0; 0 1; 0 -1; sqrt(2) sqrt(2); -sqrt(2) -sqrt(2); sqrt(2) -sqrt(2); -sqrt(2) sqrt(2)];

tree(1e6).state = [-1 -1];
tree(1).state = start;
tree(1).gCost = 0;
tree(1).fCost = heuristicCost(start, goal);
tree(1).parent = 0;
tree_index = 1;

closedSet = [];
openSet = [1 tree(1).fCost tree(1).state];

gCost = [];

iter = 0;
while ~isempty(openSet)
    iter = iter + 1;
    
    if iter == 3
        iter;
    end
    
    [current, indexInOpenSet] = get_best_from_openset(openSet);
    
    if heuristicCost(tree(current).state, goal) < 1
        path = reconstruct_path(tree, current);
        printTree(start, goal, tree, tree_index, path)
        return;
    end
    
    openSet(indexInOpenSet,:) = [];
    closedSet = [closedSet; current tree(current).state];
    
    neighbors = repmat(tree(current).state, size(A,1), 1) + A;
    
    for i = 1:size(neighbors,1)
        if is_close(neighbors(i,:), closedSet(:,2:3)) || norm(neighbors(i,:) - [20 -15]) < 12
            continue;
        end
        
        tentative_gCost = tree(current).gCost + norm(neighbors(i,:)-tree(current).state);
        
        [b, openSet_inx] = is_close(neighbors(i,:), openSet(:,3:4));
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
        
        printTree(start, goal, tree, tree_index)
        drawnow;
    end
    
    
end
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
if any(d < 1)
    b = 1;
else
    b = 0;
end

end


function printTree(start, goal, tree, tree_size, path)

if nargin == 4
    path = [];
end

figure(1)
clf
hold on
plot(start(:,1),start(:,2),'sr');
plot(goal(:,1),goal(:,2),'sr');

% Obs
x = 20; y = -15; r = 10;
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
patch(xunit, yunit,'c');

for i = 2:tree_size
    j = tree(i).parent;
    plot([tree(i).state(1) tree(j).state(1)],[tree(i).state(2) tree(j).state(2)],'k');    
end

if ~isempty(path)
    plot(path(:,1),path(:,2),'-m');
end

hold off
axis equal
axis([-40 40 -40 40]);
grid on

end