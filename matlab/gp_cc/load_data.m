function [Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source)

file = ['../../data/Cc_' data_source '_' num2str(mode)];

if strcmp(data_source, '20')
    D = load([file '.mat']);%, 'Q', 'Xtraining');%, 'Xtest1');
    Q = D.Q;
    
    Xtraining = D.Xtraining;
    switch test_num
        case 1
            Xtest = D.Xtest1.data;
            I.base_pos = D.Xtest1.base_pos;
            I.theta = D.Xtest1.theta;
            j_min = 300; j_max = size(Xtest, 1);
            I.im_min = 719;
        otherwise
            Xtest = [];            
    end
end

%%
igx = 1;
I.action_inx = Q{igx}.action_inx;
I.state_inx = Q{igx}.state_inx;
I.state_nxt_inx = Q{igx}.state_nxt_inx;
I.state_dim = length(I.state_inx);

xmax = max(Xtraining);
xmin = min(Xtraining);

for i = 1:I.state_dim
    id = [i i+I.state_dim+length(I.action_inx)];
    xmax(id) = max(xmax(id));
    xmin(id) = min(xmin(id));
end
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);
if ~isempty(Xtest)
    Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);
    Xtest = Xtest(j_min:j_max,:);
end

I.xmin = xmin;
I.xmax = xmax;

if isempty(w)
    kdtree = createns(Xtraining(:,[I.state_inx I.action_inx]), 'NSMethod','kdtree','Distance','euclidean');
else
    global W
    % W = diag([ones(1,2)*w ones(1,I.state_dim)]);
    W = diag(w);
    kdtree = createns(Xtraining(:,[I.state_inx I.action_inx]), 'Distance',@distfun);
end