Hgp = cell(8,1);
for mode = 1:8
    
    file = ['../../data/data_25_' num2str(mode)];
    
    D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest');
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
    
    Hgp{i} = tuneGP(Xtraining, I);
    
end

save('Hgb.mat','Hgb');



function Hgp = tuneGP(Xtraining, I)

C = [];
L = [];
for i = 1:100
    j = randi(size(Xtraining,1),1);
    [idx, ~] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), Xtraining(j,[I.state_inx I.action_inx]), 'K', 100);
    data_nn = Xtraining(idx,:);
    meanfunc = {'meanZero'};                    % empty: don't use a mean function
    covfunc = @covSEiso;              % Squared Exponental covariance function
    likfunc = @likGauss;              % Gaussian likelihood
    hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx));
    C = [C; hyp2.cov];
    L = [L; hyp2.lik];
end
Hgp.cov = mean(C);
Hgp.lik = mean(L);

end