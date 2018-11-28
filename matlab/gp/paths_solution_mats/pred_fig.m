clear all

data_source = '20';
test_num = 1;
mode = 8;

%%
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

Imlgp = insertText(Imlgp, [7 6], 'MLGP','fontsize',35);
imshow(Imlgp);


%%
load(['pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);

file = ['../../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);

figure(2)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),':y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% Igp = imcrop(frame.cdata, [575-250 95 200 200]); % Traj 1 & 2, x-250 for 2
Igp = imcrop(frame.cdata, [575-220 40 420 420]);

Igp = insertText(Igp, [7 6], 'EGP','fontsize',35);
imshow(Igp);


%%

Iall = [Igp; Imlgp];
Iall = border(Iall);
imshow(Iall)
% imwrite(Iall, ['test' num2str(test_num) '_20.png']);


%%
function im = border( im)

t = 8;

[rows cols n] = size(im);

im = [zeros(rows, t, n) im zeros(rows, t, n)];
im = [zeros(t, cols+2*t, n); im; zeros(t, cols+2*t, n)];

end