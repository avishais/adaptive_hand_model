function [sp, sigma_p] = prediction(kdtree, Xtraining, s, a, I, mode)

if nargin == 5
    mode = 1;
end

gprMdl = getPredictor(kdtree, Xtraining, s, a, I, mode);

sp = zeros(1, length(I.state_nxt_inx));
sigma_p = zeros(1, length(I.state_nxt_inx));
for i = 1:length(I.state_nxt_inx)
    [sp(i), sigma_p(i)] = predict(gprMdl{i}, [s a]);
end

end

function gprMdl = getPredictor(kdtree, Xtraining, x, a, I, mode)
% 
[idx, d] = knnsearch(kdtree, [x a], 'K', 1000); % Changed to 1000 after RAL paper review
data_nn = Xtraining(idx,:);

% data_nn =  diffusion_metric([x a], kdtree, Xtraining, I);

gprMdl = cell(length(I.state_nxt_inx),1);
for i = 1:length(I.state_nxt_inx)
    
    if mode == 1
        gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)),'Basis','linear','FitMethod','exact','PredictMethod','exact');
%         gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)));
    else
        if mode == 2 % Squared kernel function
            gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)),'KernelFunction','squaredexponential');
        else
            if mode == 3 % kernel parameters provided
                sigma0 = [0.0060304, 0.0037757];
                kparams0 = [3.5, 6.2];
                gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)),'KernelFunction','squaredexponential','KernelParameters',kparams0,'Sigma',sigma0(i));
            else
                if mode == 4 % Automatically find initial values for kernel parameters
                    gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)),'KernelFunction','squaredexponential','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
                        struct('AcquisitionFunctionName','expected-improvement-plus'));
                end
            end
        end
    end

end

end

% function D2 = distfun(ZI,ZJ)
% 
% global W
% 
% % W = diag([3 3 1 1 1.5 1.5]);
% if isempty(W)
%     W = diag(ones(1,size(ZI,2)));
% end    
% 
% n = size(ZJ,1);
% D2 = zeros(n,1);
% for i = 1:n
%     Z = ZI-ZJ(i,:);
%     D2(i) = Z*W*Z';
% end
%     
% end