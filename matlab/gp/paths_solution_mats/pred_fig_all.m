clear all
%%

obj = '26';
test_numd = 1;
mode = 7;

load(['./pred_all_' obj '_' num2str(test_numd) '_' num2str(mode) '_dm.mat']);

% file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
% files = dir(fullfile(file));
% IM = imread([files.folder '/' files.name]);

file_dir = ['../../../data/test_images/ca_' obj '_test' num2str(test_numd) '/'];
file = ['image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile([file_dir file]));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);
f = find_file(file, I.im_min+size(Xtest,1), files);
IM = imread([file_dir f{1}]);

figure(1)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% Imlgp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
I1 = imcrop(frame.cdata, [575-220 40 420 420]);

% Imlgp = insertText(Imlgp, [7 6], 'MLGP','fontsize',35);
imshow(I1);


%%
obj = '30';
test_numd = 3;
mode = 8;

load(['./pred_all_' obj '_' num2str(test_numd) '_' num2str(mode) '_dm.mat']);

% file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
% files = dir(fullfile(file));
% IM = imread([files.folder '/' files.name]);

file_dir = ['../../../data/test_images/ca_' obj '_test' num2str(test_numd) '/'];
file = ['image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile([file_dir file]));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);
f = find_file(file, I.im_min+size(Xtest,1), files);
IM = imread([file_dir f{1}]);

figure(1)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% Imlgp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
I2 = imcrop(frame.cdata, [575-200 40 420 420]);

% Imlgp = insertText(Imlgp, [7 6], 'MLGP','fontsize',35);
imshow(I2);

%%

obj = '36';
test_numd = 2;
mode = 7;

load(['./pred_all_' obj '_' num2str(test_numd) '_' num2str(mode) '_dm.mat']);

% file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
% files = dir(fullfile(file));
% IM = imread([files.folder '/' files.name]);

file_dir = ['../../../data/test_images/ca_' obj '_test' num2str(test_numd) '/'];
file = ['image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile([file_dir file]));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);
f = find_file(file, I.im_min+size(Xtest,1), files);
IM = imread([file_dir f{1}]);

figure(1)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% Imlgp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
I3 = imcrop(frame.cdata, [575-200 40 420 420]);

% Imlgp = insertText(Imlgp, [7 6], 'MLGP','fontsize',35);
imshow(I3);

%%

I1 = border(I1);
I2 = border(I2);
I3 = border(I3);



Iall = [I1; I2; I3];
% Iall = border(Iall);
imshow(Iall)
imwrite(Iall, ['test_all_20.png']);


%%
function im = border( im)

t = 4;

[rows cols n] = size(im);

im = [zeros(rows, t, n) im zeros(rows, t, n)];
im = [zeros(t, cols+2*t, n); im; zeros(t, cols+2*t, n)];




end