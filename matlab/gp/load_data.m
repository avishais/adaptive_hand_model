function [Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source)

if nargin==1
    UseToyData = false;
    test_num = 1;
    w = 1;
    data_source = '25';
end
if nargin==2
    test_num = 1;
    UseToyData = false;
    data_source = '25';
end
if nargin==3
    data_source = '25';
end

UseToyData = 0;

file = ['../../data/Ca_' data_source '_' num2str(mode)];

%% Toy Data

if strcmp(data_source, 'toy')
    Xtraining = load('../../data/toyData.db');
    Xtest = load('../../data/toyDataPath.db');
    j_min = 1; j_max = size(Xtest,1);
    I.action_inx = 3:4;
    I.state_inx = 1:2;
    I.state_nxt_inx = 5:6;
    I.state_dim = length(I.state_inx);
end
%%
if strcmp(data_source, '25')
    D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest1','Xtest2', 'Xtest3');
    Q = D.Q;
    
    Xtraining = D.Xtraining;
    switch test_num
        case 1
            Xtest = D.Xtest1.data;
            I.base_pos = D.Xtest1.base_pos;
            I.theta = D.Xtest1.theta;
            j_min = 350; j_max = 680;%size(Xtest, 1);
        case 2
            Xtest = D.Xtest2.data;
            I.base_pos = D.Xtest2.base_pos;
            I.theta = D.Xtest2.theta;
            j_min = 1; j_max = size(Xtest,1);
           c
        case 3
            Xtest = D.Xtest3.data;
            I.base_pos = D.Xtest3.base_pos;
            I.theta = D.Xtest3.theta;
            j_min = 1038; j_max = size(Xtest,1);
            I.im_min = 1419;
        otherwise
            Xtest = [];            
    end
end

if strcmp(data_source, '20')
    D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest1','Xtest2', 'Xtest3');
    Q = D.Q;
    
    Xtraining = D.Xtraining;
    switch test_num
        case 1
            Xtest = D.Xtest1.data;
            I.base_pos = D.Xtest1.base_pos;
            I.theta = D.Xtest1.theta;
            j_min = 700; j_max = size(Xtest, 1);
            I.im_min = 719;
        case 2
            Xtest = D.Xtest2.data;
            I.base_pos = D.Xtest2.base_pos;
            I.theta = D.Xtest2.theta;
            j_min = 700; j_max = size(Xtest,1);
            I.im_min = 2427+j_min;
        case 3
            Xtest = D.Xtest3.data;
            I.base_pos = D.Xtest3.base_pos;
            I.theta = D.Xtest3.theta;
            j_min = 1; j_max = size(Xtest,1);
            I.im_min = 4743;
        otherwise
            Xtest = [];            
    end
end

%%
if strcmp(data_source, 'all')
    D = load([file '.mat'], 'Q', 'Xtraining', 'Xtest_15_1','Xtest_15_2','Xtest_15_3', 'Xtest_30_1','Xtest_30_2','Xtest_11_1','Xtest_31_1','Xtest_31_2','Xtest_30_3','Xtest_15_4','Xtest_45_1','Xtest_26_1','Xtest_36_1','Xtest_36_2');
    Q = D.Q;
    
    Xtraining = D.Xtraining; 
    switch test_num
        case 1
            Xtest = D.Xtest_15_1.data;
            I.base_pos = D.Xtest_15_1.base_pos;
            I.theta = D.Xtest_15_1.theta;
            j_min = 362; j_max = size(Xtest, 1);
            I.im_min = 461+j_min-1;
            I.test_data_source = {'15', '1', '1'};
        case 2
            Xtest = D.Xtest_15_2.data;
            I.base_pos = D.Xtest_15_2.base_pos;
            I.theta = D.Xtest_15_2.theta;
            j_min = 1; j_max = size(Xtest,1);
            I.im_min = 455;
            I.test_data_source = {'15', '2', '2'};
        case 3
            Xtest = D.Xtest_15_3.data;
            I.base_pos = D.Xtest_15_3.base_pos;
            I.theta = D.Xtest_15_3.theta;
            j_min = 50; j_max = size(Xtest,1);
            I.im_min = 1840+j_min-1;
            I.test_data_source = {'15','3', '3'};
        case 4
            Xtest = D.Xtest_30_1.data;
            I.base_pos = D.Xtest_30_1.base_pos;
            I.theta = D.Xtest_30_1.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 630;
            I.test_data_source = {'30','1','1'};
        case 5
            Xtest = D.Xtest_30_2.data;
            I.base_pos = D.Xtest_30_2.base_pos;
            I.theta = D.Xtest_30_2.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 481;
            I.test_data_source = {'30','2','2'};
        case 6
            Xtest = D.Xtest_11_1.data;
            I.base_pos = D.Xtest_11_1.base_pos;
            I.theta = D.Xtest_11_1.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 660;
            I.test_data_source = {'11','1', '3'}; 
        case 7 % Acrylic paint
            Xtest = D.Xtest_31_1.data;
            I.base_pos = D.Xtest_31_1.base_pos;
            I.theta = D.Xtest_31_1.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 68;
            I.test_data_source = {'31','1','3'};
        case 8 % Acrylic paint
            Xtest = D.Xtest_31_2.data;
            I.base_pos = D.Xtest_31_2.base_pos;
            I.theta = D.Xtest_31_2.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 1846;
            I.test_data_source = {'31','2','3'};
       case 9 % Glue stick
            Xtest = D.Xtest_30_3.data;
            I.base_pos = D.Xtest_30_3.base_pos;
            I.theta = D.Xtest_30_3.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 43;
            I.test_data_source = {'30','3','3'};
            lim1 = 6; lim2 = 28;
        case 10 % Thick sharpie
            Xtest = D.Xtest_15_4.data;
            I.base_pos = D.Xtest_15_4.base_pos;
            I.theta = D.Xtest_15_4.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 53;
            I.test_data_source = {'15','4','3'};
        case 11 % Cyl 45
            Xtest = D.Xtest_45_1.data;
            I.base_pos = D.Xtest_45_1.base_pos;
            I.theta = D.Xtest_45_1.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 71;
            I.test_data_source = {'45','1','3'};
        case 12 % Butter can
            Xtest = D.Xtest_26_1.data;
            I.base_pos = D.Xtest_26_1.base_pos;
            I.theta = D.Xtest_26_1.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 59;
            I.test_data_source = {'26','1','3'};
            lim1 = 8; lim2 = 28;
        case 13 % Hair-spary
            Xtest = D.Xtest_36_1.data;
            I.base_pos = D.Xtest_36_1.base_pos;
            I.theta = D.Xtest_36_1.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 1996;
            I.test_data_source = {'36','1','3'};
        case 14 % Hair-spary - same traj as previous but up-side down
            Xtest = D.Xtest_36_2.data;
            I.base_pos = D.Xtest_36_2.base_pos;
            I.theta = D.Xtest_36_2.theta;
            j_min = 1; j_max = size(Xtest, 1);
            I.im_min = 3856;
            I.test_data_source = {'36','2','3'};
            lim1 = 10; lim2 = 28;
    end
    figure(1)
    subplot(121)
    plot(Xtraining(:,1),'.')
    
    c = mean(Xtest(:,1));
    Xtraining(abs(Xtraining(:,1)-c) < lim1,:)=[];
    Xtraining = Xtraining(abs(Xtraining(:,1)-c) <= lim2, 1:end);
    subplot(122)
    plot(Xtraining(:,1),'.')
    Xtest = Xtest(:,2:end);
end

%%
igx = 1;
I.action_inx = Q{igx}.action_inx;
I.state_inx = Q{igx}.state_inx;
I.state_nxt_inx = Q{igx}.state_nxt_inx;
I.state_dim = length(I.state_inx);

xmax = max(Xtraining);
xmin = min(Xtraining);
% if mode==11 % So it is possible to extrapolate on the objects diameter
%     xmin(I.state_dim+length(I.action_inx)) = 0;
% end

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

clear Q D
