function safe = classifier(state_action, kdtree, bound, r)

id = rangesearch(kdtree, state_action, r); id = id{1};

if length(id) > bound
    safe = 1;
else
    safe = 0;
end

end


