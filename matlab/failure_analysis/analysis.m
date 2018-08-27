clear all

addpath('../gp/');

mode = 8;
load(['class_data_ ' num2str(mode) '.mat']);

switch mode
    case 1
        w = [];%[1 1 1 1];
        r = sqrt(0.01);
    case 5
        w = [];%[1 1 1 1 1 1];
        r = sqrt(0.15);
    case 8
%         w = [1 1 1 1 1 1 1 1];
%         r = 0.15;
        w = [];
        r = sqrt(0.15);
end
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, 1, '20');

n = size(data,1);
data = (data-repmat(I.xmin([I.state_inx I.action_inx]), n, 1))./repmat(I.xmax([I.state_inx I.action_inx])-I.xmin([I.state_inx I.action_inx]), n, 1);

%%

NN = zeros(n,1);
for i = 1:n
    
    sa = data(i,:);
    
    id = rangesearch(kdtree, sa, r); id = id{1};
    
    NN(i) = length(id);    
end

%%
sr = 0;
for b = 10:10:30000
    approx_labels = NN > b;
    success_rate = sum(labels==approx_labels)/n * 100;
    if success_rate > sr
        sr = success_rate;
        b_max = b;
    end
end

approx_labels = NN > b_max;

% [max(NN(1:16)) min(NN(17:end))]

disp(['Success rate: ' num2str(sr) ' with bound ' num2str(b_max)]);

success_rate_drop = sum(labels(1:18)==approx_labels(1:18))/18 * 100;
success_rate_sing = sum(labels(19:30)==approx_labels(19:30))/12 * 100;
success_rate_fail = sum(labels(1:30)==approx_labels(1:30))/30 * 100;
success_rate_normal = sum(labels(31:60)==approx_labels(31:60))/30 * 100;


disp(['Drop success rate: ' num2str(success_rate_drop)]); 
disp(['Singularity success rate: ' num2str(success_rate_sing)]); 
disp(['Normal success rate: ' num2str(success_rate_normal)]);
disp(['Fail success rate: ' num2str(success_rate_fail)]); 



% bar(NN);

% histfit(NN);

figure(1)
clf
bar([sort(NN(31:60)) sort(NN(1:30))]);
ylim([0 5e4]);
set(gca,'XTick',[])
xlabel('Test states','fontsize',16');
ylabel('Number of nearest neighbors','fontsize',16');
legend({'Normal states','Failure states'},'location','northwest','FontSize',14);
print(['nnclass.png'],'-dpng','-r150');

