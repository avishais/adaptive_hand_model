clear all
warning('off','all')

px2mm = 0.2621;

with_figure = false;

data_source = '20';
test_num = 1;
mode = 8;
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
        w = [6 6 3 3 1 1 1 1];
end
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source);

Sr = Xtest;


%% open loop
SRI = zeros(size(Sr,1), 2);
for i = 1:size(Sr,1)
    SRI(i,:) = project2image(Sr(i,1:2), I);
end
if with_figure
	file = ['../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
	files = dir(fullfile(file));
	IM = imread([files.folder '/' files.name]);
end
    
figure(2)
clf
if with_figure
	imshow(IM);
end
hold on
plot(SRI(:,1),SRI(:,2),'-b','linewidth',3,'markerfacecolor','y');
axis equal

tic;
s = Sr(1,I.state_inx);
S = zeros(size(Sr,1), I.state_dim);
SI = zeros(size(Sr,1), 2);
S(1,:) = s;
SI(1,:) = project2image(s(1:2), I);
loss = 0;
for i = 1:size(Sr,1)-1
    a = Sr(i, I.action_inx);
    disp(['Step: ' num2str(i) ', action: ' num2str(a)]);
    [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);
    S(i+1,:) = s;
    
    SI(i+1,:) = project2image(s(1:2), I);
    
    if ~mod(i, 10)
        plot(SI(1:i,1),SI(1:i,2),'.-m');
        drawnow;
        disp(['mse = ' num2str(MSE(SRI(1:i,1:2), SI(1:i,1:2)) * px2mm)]);
    end
end
S = S(1:i+1,:);
hold off

disp(toc)

disp(['mse = ' num2str(MSE(SRI, SI) * px2mm)]);

save(['./paths_solution_mats/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_test.mat'],'data_source','I','loss','mode','S','SI','Sr','SRI','test_num','w','Xtest');

%%

figure(1)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on
plot(S(:,1),S(:,2),'.-r');
plot(S(1,1),S(1,2),'or','markerfacecolor','r');
hold off
axis equal
legend('ground truth','predicted path');
title(['open loop - ' num2str(mode) ', MSE: ' num2str(loss)]);
disp(['Loss: ' num2str(loss)]);

%%
load(['./paths_solution_mats/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);
if with_figure
	file = ['../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
	files = dir(fullfile(file));
	IM = imread([files.folder '/' files.name]);
end

figure(2)
clf
if with_figure
	imshow(IM);
end
hold on
plot(SRI(:,1),SRI(:,2),'-y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
% frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% frame.cdata = imcrop(frame.cdata, [50 80 431+432 311]);
% imshow(frame.cdata);
% imwrite(frame.cdata, ['test' num2str(test_num) '_' num2str(data_source) '.png']);

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

