function sp = prediction(Xtraining, s, a, I, mode)

if nargin == 4
    mode = 1;
end

gprMdl = getPredictor(Xtraining, s, a, I, mode);

sp = zeros(1, length(I.state_nxt_inx));
for i = 1:length(I.state_nxt_inx)
    sp(i) = predict(gprMdl{i}, [s a]);
end

end

function gprMdl = getPredictor(Xtraining, x, a, I, mode)

[idx, ~] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), [x a], 'K', 100);
data_nn = Xtraining(idx,:);

gprMdl = cell(length(I.state_nxt_inx),1);
for i = 1:length(I.state_nxt_inx)
    
    if mode == 1
        gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)),'Basis','linear','FitMethod','exact','PredictMethod','exact');
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