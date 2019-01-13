clear all

data_source = '20';

%%
test_num = 1;
mode = 8;

load(['./beforeDM/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);
% load(['./pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_dm.mat']);
% load(['./pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);


file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);

figure(1)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% Imlgp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
Imlgp = imcrop(frame.cdata, [575-10 90 200 200]);

Imlgp1 = insertText(Imlgp, [7 165], 'Test traj. 1','fontsize',20, 'textcolor','w', 'boxopacity', 0);
imshow(Imlgp1);


%%
test_num = 2;
mode = 8;

load(['./beforeDM/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);
% load(['./pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_dm.mat']);

file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);

figure(1)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
Imlgp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
% Imlgp = imcrop(frame.cdata, [575-220 40 420 420]);

Imlgp2 = insertText(Imlgp, [7 165], 'Test traj. 2','fontsize',20, 'textcolor','w', 'boxopacity', 0);
imshow(Imlgp2);

%%

test_num = 3;
mode = 8;

% load(['./beforeDM/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);
load(['./pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_dm.mat']);

file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);

figure(1)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% Imlgp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
Imlgp = imcrop(frame.cdata, [575-220 40 420 420]);
Imlgp3 = imresize(Imlgp, [size(Imlgp1,1), size(Imlgp1,2)]);

Imlgp3 = insertText(Imlgp3, [7 165], 'Test traj. 3','fontsize',20, 'textcolor','w', 'boxopacity', 0);
imshow(Imlgp3);

%%

Iall = [border(Imlgp1) border(Imlgp2) border(Imlgp3)];
Iall(:,[201:211 417:422] ,:) = [];
% Iall = border(Iall);
imshow(Iall)
% imwrite(Iall, ['test_20_mlgp.png']);


%%
function im = border( im)

t = 5;

[rows cols n] = size(im);

im = [zeros(rows, t, n) im zeros(rows, t, n)];
im = [zeros(t, cols+2*t, n); im; zeros(t, cols+2*t, n)];

end