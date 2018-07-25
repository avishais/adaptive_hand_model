function sp = prediction(Xtraining, s, a, I)

gprMdl = getPredictor(Xtraining, s, a, I);

sp = zeros(1, I.state_dim);
for i = 1:I.state_dim
    sp(i) = predict(gprMdl{i}, [s a]);
end

end

function gprMdl = getPredictor(Xtraining, x, a, I)

[idx, ~] = knnsearch(Xtraining(:,[I.state_inx I.action_inx]), [x a], 'K', 50);
data_nn = Xtraining(idx,:);

gprMdl = cell(2,1);
for i = 1:I.state_dim
    gprMdl{i} = fitrgp(data_nn(:,[I.state_inx I.action_inx]), data_nn(:,I.state_nxt_inx(i)),'Basis','linear','FitMethod','exact','PredictMethod','exact');
end

end