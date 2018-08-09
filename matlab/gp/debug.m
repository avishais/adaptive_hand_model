clear all

mode = 1;
file = ['../../data/data_25_' num2str(mode)];

D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest','Xtest2');
Q = D.Q;
I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

Xtraining = D.Xtraining; 

xmax = max(Xtraining); 
xmin = min(Xtraining); 
for i = 1:I.state_dim
    id = [i i+I.state_dim+length(I.action_inx)];
    xmax(id) = max(xmax(id));
    xmin(id) = min(xmin(id));
end
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);

clear Q D

%%

a = [1 1];
x = [0.5406 0.2978];
[idx, d] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), [x a], 'K', 100);

data_nn = Xtraining(idx,:);

dim = 1;
gprMdl = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(dim)));

[sp, sigma_p] = predict(gprMdl, [x a]);

sp