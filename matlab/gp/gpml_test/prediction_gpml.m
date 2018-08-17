function [sp, sigma] = prediction_gpml(Xtraining, s, a, I, Hgp)

if nargin == 4 
    Hgp.cov = [-0.6211,-0.6342];
    Hgp.lik = -8.7993;
end

% Take data in the vicinity if the query point
[idx, ~] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), [s a], 'K', 100);
% [idx, d] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), [s a], 'K', 100, 'Distance',@distfun);
data_nn = Xtraining(idx,:);

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

if isempty(Hgp)
    hyp = struct('mean', [], 'cov', [0, 0], 'lik', -1);
    hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx));
else
    
    hyp = struct('mean', [], 'cov', Hgp.cov, 'lik', Hgp.lik);
end

[mu, s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx), [s a]);

sp = mu;
sigma = s2.^0.5;

end

function D2 = distfun(ZI,ZJ)

% W = diag([3 3 1 1 1.5 1.5]);
W = diag([7 7 0.51 0.51 1 1]);

n = size(ZJ,1);
D2 = zeros(n,1);
for i = 1:n
    Z = ZI-ZJ(i,:);
    D2(i) = Z*W*Z';
end
    
end