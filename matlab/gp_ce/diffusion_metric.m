function data_nn = diffusion_metric(sa, kdtree, Xtraining, I)

[idx, ~] = knnsearch(kdtree, sa, 'K', 1001);
data = Xtraining(idx,:);

if I.state_dim == 6
    dim = 3;
end
if I.state_dim == 4
    dim = 2;
else
    dim = 3;
end

data_reduced = dr_diffusionmap(data(:,[I.state_inx I.action_inx]), dim);
sa_reduced_closest = data_reduced(1,:);
data_reduced = data_reduced(2:end,:);

idx_new = knnsearch(data_reduced, sa_reduced_closest, 'K', 100);

data_nn = data(idx_new,:);