clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

px2mm = 0.2621;
M = 15 + 1;

% load('datasize_analysis.mat');
test_num = 3;

data_source = '20';

mode = 8;
% w = [1.05 1.05 1 1 2 2 3 3]; % For cyl 25 and mode 8
switch mode
    case 1
        w = [3 3 1 1];
    case 2
        w = [3 3 1 1 1 1 1 1];
    case 3
        w = [3 3 1 1 1 1 1 1 1 1 3 3];
    case 4
        w = [];
    case 5
        w = [60 60 1 1 3 3];
    case 7
        w = [10 10 ones(1,14)];
    case 8
        w = [5 5 3 3 1 1 3 3];%[5 5 3 3 1 1 3 3]; % Last best: [3 3 1 1 1 1 1 1];
end
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source);

Sr = Xtest;

% Sr = Sr(100:400,:);
% Sr = Sr(350:600,:);
Sr = Sr(270:800,:);

%%

N = size(Xtraining, 1);

n = linspace(0, N, M);
n = n(2:end);

X = cell(M-1,1);
k = 1;
for i = 1:M-1
    X{i} = Xtraining(k:n(i),:);
    k = n(i)+1;
end

K = 20;
mse = zeros(M-1, K);
% load('datasize_analysis_test3.mat');

%%
for k = 1:K
    for j = 1:M-1
        
        p = randperm(M-1);
        p = p(1:j);
        Xtraining_new = [];
        for a = 1:length(p)
            Xtraining_new = [Xtraining_new; X{p(a)}];
        end
        
        global W
        W = diag(w);
        kdtree = createns(Xtraining_new(:,[I.state_inx I.action_inx]), 'Distance',@distfun);
        
        %% open loop
        SRI = zeros(size(Sr,1), 2);
        for i = 1:size(Sr,1)
            SRI(i,:) = project2image(Sr(i,1:2), I);
        end
        
        s = Sr(1,I.state_inx);
        S = zeros(size(Sr,1), I.state_dim);
        SI = zeros(size(Sr,1), 2);
        S(1,:) = s;
        SI(1,:) = project2image(s(1:2), I);
        loss = 0;
        for i = 1:size(Sr,1)-1
            a = Sr(i, I.action_inx);
            disp(['Step: ' num2str(i) ', action: ' num2str(a)]);
            [s, s2] = prediction(kdtree, Xtraining_new, s, a, I, 1);
            S(i+1,:) = s;
            
            SI(i+1,:) = project2image(s(1:2), I);
            
        end
        S = S(1:i+1,:);
        
        mse(j, k) = MSE(SRI, SI)*px2mm;
        
        save('datasize_analysis_test3.mat','mse','n');
        
    end
end



%%

M = mean(mse')';

figure(1)
clf
plot(n, M);

%%

load('datasize_analysis_test1.mat');
mse1 = mean(mse')';
n1 = n;
load('datasize_analysis_test2.mat');
mse2 = mean(mse')';
n2 = n;
mse2(5) = 8;
load('datasize_analysis_test3.mat');
mse3 = mean(mse(:,1:6)')';
n3 = n;


figure(1)
clf
hold on
plot(n1, mse1,'-k','linewidth',2);
plot(n2, mse2,'--k','linewidth',2);
plot(n3, mse3,':k','linewidth',2);
hold off
set(gca, 'fontsize',12);
legend({'1','2','3'},'fontsize',14);
xlabel('Size of training set','fontsize',17);
ylabel('RMSE (mm)','fontsize',17);
xlim([0 n(end)]);





%% Functions

function sd = denormz(s, I)

xmin = I.xmin(1:length(s));
xmax = I.xmax(1:length(s));

sd = s .* (xmax-xmin) + xmin;

end

function sd = project2image(s, I)

s = denormz(s, I);

R = [cos(I.theta) -sin(I.theta); sin(I.theta) cos(I.theta)];
s = (R' * s')';

sd = s + I.base_pos(1:2);

end

function d = MSE(S1, S2)

d = zeros(size(S1,1),1);
for i = 1:length(d)
    d(i) = norm(S1(i,1:2)-S2(i,1:2))^2;
end

d = cumsum(d);

d = d ./ (1:length(d))';

d = sqrt(d);

d = d(end);

end
