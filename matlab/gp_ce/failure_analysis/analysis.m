clear all

addpath('../');

mode = 5;
load(['class_data_ ' num2str(mode) '.mat']);

% w = [1 1 1 1 10 10];
w = [];
r = 0.9;
[~, ~, kdtree, I] = load_data(mode, w, 1, '20');

n = size(data,1);
% data = (data-repmat(I.xmin([I.state_inx I.action_inx]), n, 1))./repmat(I.xmax([I.state_inx I.action_inx])-I.xmin([I.state_inx I.action_inx]), n, 1);

GP = gp_class(5);
%%

NN = zeros(n,1);
for i = 1:n
    
    sa = data(i,:);
    
%     id = rangesearch(kdtree, sa, r); id = id{1};
%     NN(i) = length(id);    
    
    NN(i) = GP.getNN(sa(1:4), sa(5:6), r);
    
%     NN(i) = diffusion_metric_nn(sa, kdtree, Xtraining, I, r);

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

disp(['Success rate: ' num2str(sr) ' with bound ' num2str(b_max)]);

success_rate_drop = sum(labels(1:50)==approx_labels(1:50))/50 * 100;
success_rate_normal = sum(labels(51:75)==approx_labels(51:75))/25 * 100;


disp(['Drop success rate: ' num2str(success_rate_drop)]); 
disp(['Normal success rate: ' num2str(success_rate_normal)]);

% bar(NN);

% histfit(NN);

figure(1)
clf
bar([sort(NN(51:75)); sort(NN(1:50))]');
% ylim([0 5e4]);
set(gca,'XTick',[])
xlabel('Test states','fontsize',16');
ylabel('Number of nearest neighbors','fontsize',16');
% legend({'Normal states','Failure states'},'location','northwest','FontSize',14);
print(['nnclass.png'],'-dpng','-r150');

