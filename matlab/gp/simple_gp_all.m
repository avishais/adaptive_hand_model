clear all
for test_num = [15 16 17]
% 
for mode = [1 2 3 4 5 7 8]
warning('off','all')

% if ~exist('is_nm','var')
%     clear all
    
% mode = 8;
% end

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

% test_num = 16;
% w = [1 1 10 10 10 10 1 1 1];
w = [];
% switch mode
%     case 1
%         w = [3 3 1 1];
%     case 2
%         w = [3 3 1 1 1 1 1 1];
%     case 3
%         w = [3 3 1 1 1 1 1 1 1 1 3 3];
%     case 4
%         w = [];
%     case 5
%         w = [];%[60 60 1 1 3 3];
%     case 7
%         w = [10 10 ones(1,14)];
%     case 8
%         w = [];%[5 5 3 3 1 1 3 3]; % Last best: [3 3 1 1 1 1 1 1];
%     case 11
%         w = [];
% end
data_source = 'all';
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source);

Sr = Xtest;

%% open loop
SRI = zeros(size(Sr,1), 2);
for i = 1:size(Sr,1)
    SRI(i,:) = project2image(Sr(i,1:2), I);
end
file = ['../../data/test_images/ca_' I.test_data_source{1} '_test' I.test_data_source{2} '/image_test' I.test_data_source{3} '_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);
 
figure(2)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),'-b','linewidth',3,'markerfacecolor','y');
axis equal

offset = 0;

tic;
s = Sr(1+offset,I.state_inx);
S = zeros(size(Sr,1), I.state_dim);
SI = zeros(size(Sr,1), 2);
S(1,:) = s;
SI(1,:) = project2image(s(1:2), I);
loss = 0;
cyl = s(I.state_dim);
for i = 1:size(Sr,1)-1
    a = Sr(i+offset, I.action_inx);
    disp(['Step: ' num2str(i) ', action: ' num2str(a)]);
    [s, s2] = prediction_mlgp(kdtree, Xtraining, s, a, I, 1);
    S(i+1,:) = s;
    loss = loss + norm(s - Sr(i+1, I.state_nxt_inx));
    
    SI(i+1,:) = project2image(s(1:2), I);
    
    if ~mod(i, 10)
        plot(SI(1:i,1),SI(1:i,2),'.-m');
        drawnow;
    end
end
S = S(1:i+1,:);
hold off

loss = loss / size(Sr,1);
disp(toc)

save(['./paths_solution_mats/pred_' data_source '_' I.test_data_source{1} '_' I.test_data_source{2} '_' num2str(mode) '_dm.mat'],'data_source','I','loss','mode','S','SI','Sr','SRI','test_num','w','Xtest');


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
% load(['./paths_solution_mats/pred_' data_source '_' I.test_data_source{1} '_' I.test_data_source{2} '_' num2str(mode) '.mat']);

file = ['../../data/test_images/ca_' I.test_data_source{1} '_test' I.test_data_source{2} '/image_test' I.test_data_source{3} '_' num2str(I.im_min+size(Xtest,1)) '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);

figure(2)
clf
imshow(IM);
hold on
plot(SRI(:,1),SRI(:,2),'-y','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-c','linewidth',3);
hold off
% frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
% frame.cdata = imcrop(frame.cdata, [50 80 431+432 311]);
% imshow(frame.cdata);

% imwrite(frame.cdata, ['test' num2str(test_num) '_' num2str(data_source) '.png']);
end
end
%% Closed loop

% figure(2)
% clf
% hold on
% plot(Sr(:,1),Sr(:,2),'o-b','linewidth',2,'markerfacecolor','k');
% 
% Sp = zeros(size(Sr,1),I.state_dim);
% for i = 1:size(Sr,1)
%     disp(['Step: ' num2str(i)]);
%     
%     s = Sr(i,I.state_inx);
%     a = Sr(i, I.action_inx);
%     sr = Sr(i,I.state_nxt_inx);
%     sp = prediction(kdtree, Xtraining, s, a, I);
%     Sp(i,:) = sp;
% %     drawnow;
% 
%     if ~mod(i, 10)
%         for j = 1:i
%             plot([Sr(j,1) Sp(j,1)],[Sr(j,2) Sp(j,2)],'.-r');
%         end
%         drawnow;
%     end
% end
% 
% hold off

%%
% figure(2)
% clf
% hold on
% plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');
% for i = 1:size(Sr,1)
%         plot([Sr(i,1) Sp(i,1)],[Sr(i,2) Sp(i,2)],'.-');
% end
% plot(Sr(1,1),Sr(1,2),'or','markerfacecolor','m');
% hold off
% axis equal
% legend('original path');
% title('Closed loop');


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
