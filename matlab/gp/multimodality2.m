clear all
warning('off','all')

UseToyData = false;

K = 8;%[1 5 8];
n = 100;
G = zeros(n, length(K));
for i = 1:length(K)
    mode = K(i);
    switch mode
        case 1
            w = [];
        case 5
            w = [60 60 1 1 3 3];
        case 8
            w = [5 5 3 3 1 1 3 3]; % Last best: [5 5 3 3 1 1 3 3];
    end
    test_num = 1;
    [Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, '20');
    
    %%
    
    k = randi(size(Xtraining,1));
    
    xa = Xtraining(k, [I.state_inx I.action_inx]);

    [idx, d] = knnsearch(kdtree, xa, 'K', n);
    
    dnn = Xtraining(idx(1:end),:);
    
    N = zeros(size(dnn,1),1);
    for j = 1:size(dnn,1)
        v = dnn(j, I.state_nxt_inx) - dnn(j, I.state_inx);
        G(j,i) = rad2deg(atan2(v(2),v(1)));
        N(j) = norm(v);
    end
    
    G(G(:,i)<0,i) = G(G(:,i)<0,i) + 360;
end

%%

figure(1)
clf
hist(G,100);
xlim([0 360]);
title(['Feature conf. ' num2str(mode)]);

