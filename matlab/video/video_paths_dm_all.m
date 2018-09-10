clear all

data_source = 'all';
obj = 26;%36;30;26;
test_numd = 1;%2;%1;
mode = 7;%7;

speed = 5;
record = 1;

switch test_numd
    case 1
        xd = 350;
    case 2
        xd = 300;
    case 3
        xd = 310;
end



load(['../gp/paths_solution_mats/pred_' data_source '_' num2str(obj) '_' num2str(test_numd) '_' num2str(mode) '_dm.mat']);
% load(['../gp/paths_solution_mats/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_dm.mat']);

%%

if record
    writerObj = VideoWriter(['traj_' num2str(obj) '_' num2str(test_numd) '_all_dm.avi']);
    writerObj.FrameRate = 60;
    open(writerObj);
end

of = 0;
j = 1+of;
for i = I.im_min+of:speed:I.im_min+size(Xtest,1)-1
    disp(['Step DM ' num2str(i-I.im_min)]);
    
    file_dir = ['../../data/test_images/ca_' num2str(obj) '_test' num2str(test_numd) '/'];
    file = ['image_test3_' num2str(i) '*.jpg'];
    files = dir(fullfile([file_dir file]));
    files = struct2cell(files)';
    files = sortrows(files, 1);
    files = files(:,1);
    f = find_file(file, i, files);
    IM = imread([file_dir f{1}]);
    
    figure(1)
    clf
    imshow(IM);
    hold on
    plot(SRI(1:j,1),SRI(1:j,2),':y','linewidth',3,'markerfacecolor','y');
    plot(SI(1:j,1),SI(1:j,2),'-c','linewidth',3);
    hold off
    frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
    Imlgp = imcrop(frame.cdata, [xd 50 500 500]); % Traj 1 & 2, x-250 for 2
    
    Imlgp = insertText(Imlgp, [7 430], 'Butter can','fontsize',35);
%     imshow(Imlgp);
    
    if record
        writeVideo(writerObj, Imlgp);
    end
    
    j = j + speed;
end


if record
    close(writerObj); % Saves the movie.
end

