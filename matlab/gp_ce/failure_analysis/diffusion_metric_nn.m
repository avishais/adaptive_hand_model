function N = diffusion_metric_nn(sa, kdtree, Xtraining, I, r)

[idx, ~] = knnsearch(kdtree, sa, 'K', 2001);
data = Xtraining(idx,:);

data_reduced = dr_diffusionmap(data(:,[I.state_inx I.action_inx]), 2);
sa_reduced_closest = data_reduced(1,:);
data_reduced = data_reduced(2:end,:);

idx_new = rangesearch(data_reduced, sa_reduced_closest, r);

N = length(idx_new{1});