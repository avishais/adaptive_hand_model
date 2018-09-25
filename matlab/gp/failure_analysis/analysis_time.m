clear all

addpath('../gp/');
modes = [1 5 8];
for k = 1:length(modes)
    mode = modes(k);
    load(['class_data_ ' num2str(mode) '.mat']);
    
    switch mode
        case 1
            w = [];
            r = sqrt(0.01);
            bound = 2360;
        case 5
            w = [];
            r = sqrt(0.15);
            bound = 12210;
        case 8
            w = [];
            r = sqrt(0.15);
            bound = 8560;
    end
    [Xtraining, Xtest, kdtree, I] = load_data(mode, w, 1, '20');
    
    n = size(data,1);
    data = (data-repmat(I.xmin([I.state_inx I.action_inx]), n, 1))./repmat(I.xmax([I.state_inx I.action_inx])-I.xmin([I.state_inx I.action_inx]), n, 1);
    
    %%
    
    Q = Q(1:30);
    
    h = 240; % horizon
    D = zeros(h, length(Q));
    for i = 1:length(Q)
        disp([k i]);
        M = Q{i}(end-h+1:end,[I.state_inx I.action_inx]);
        M = (M-repmat(I.xmin([I.state_inx I.action_inx]), h, 1))./repmat(I.xmax([I.state_inx I.action_inx])-I.xmin([I.state_inx I.action_inx]), h, 1);
        
        for j = 1:size(M,1)
            D(j, i) = classifier(M(j,:), kdtree, bound, r);
        end
        Q{i} = M;
    end
    
    %%
    Ds{k}  = 1-sum(D')'./size(D,2);
    T = (1:h) / 15;
    
end

%%
windowSize = 25; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;


for i = 1:length(Ds)
    F(:,i) = filter(b,a,Ds{i});
end


figure(2)
clf
hold on
for i = 1:length(Ds)
    plot(T, F);
end
hold off
legend('1','2','3');


%%
save('D.mat');