function [Xtraining, Xtest, kdtree, I] = load_data(mode, test_num, UseToyData)

if nargin==1
    UseToyData = false;
    test_num = 1;
end
if nargin==2
    UseToyData = false;
end

file = ['../../data/Ca_25_' num2str(mode)];

D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest1','Xtest2', 'Xtest3');
Q = D.Q;
I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

if UseToyData
    Xtraining = load('../../data/toyData.db');
    Xtest = load('../../data/toyDataPath.db');
    j_min = 1; j_max = size(Xtest,1);
else
    Xtraining = D.Xtraining; 
    if test_num==1
        Xtest = D.Xtest1.data;
        I.base_pos = D.Xtest1.base_pos;
        I.theta = D.Xtest1.theta;
        j_min = 270; j_max = 600;%size(Xtest, 1);
    else
        if test_num==2
            Xtest = D.Xtest2.data;
            I.base_pos = D.Xtest2.base_pos;
            I.theta = D.Xtest2.theta;
            j_min = 1; j_max = size(Xtest,1);
        else
            if test_num==3
                Xtest = D.Xtest3.data;
                I.base_pos = D.Xtest3.base_pos;
                I.theta = D.Xtest3.theta;
                j_min = 1; j_max = size(Xtest,1);
            end
        end
    end
end

xmax = max(Xtraining); 
xmin = min(Xtraining); 
for i = 1:I.state_dim
    id = [i i+I.state_dim+length(I.action_inx)];
    xmax(id) = max(xmax(id));
    xmin(id) = min(xmin(id));
end
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);
Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

Xtest = Xtest(j_min:j_max,:);

I.xmin = xmin;
I.xmax = xmax;

global W
% W = diag([10 10 2 2 1 1]);
W = diag([ones(1,2)*1 ones(1,I.state_dim)]);

kdtree = createns(Xtraining(:,[I.state_inx I.action_inx]),'Distance',@distfun);

clear Q D