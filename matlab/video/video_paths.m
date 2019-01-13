clear all

data_source = '20';
test_num = 2;
% mode = 8;

speed = 5;
record = 0;

switch test_num
    case 1
        xd = 350;
    case 2
        xd = 200;
    case 3
        xd = 310;
end

%%
k = 6;
for mode = 8%[1 2 3 4 5 8 7]
    
    if mode == 8
        load(['../gp/paths_solution_mats/beforeDM/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '.mat']);
    else
        load(['../gp/paths_solution_mats/pred_' data_source '_' num2str(mode) '_' num2str(test_num) '_dm.mat']);
    end
    
    
    if record
        writerObj = VideoWriter(['traj_' num2str(k) '_' num2str(test_num) '_dm.avi']);
        writerObj.FrameRate = 60;
        open(writerObj);
    end
    
    j = 1;
    for i = I.im_min:speed:I.im_min+size(Xtest,1)
        disp(['Step DM ' num2str(i-I.im_min)]);
        
        file = ['../../data/test_images/ca_' num2str(data_source) '_test' num2str(test_num) '/image_test3_' num2str(i) '*.jpg'];
        files = dir(fullfile(file));
        IM = imread([files.folder '/' files.name]);
        
        figure(1)
        clf
        imshow(IM);
        hold on
        plot(SRI(1:j,1),SRI(1:j,2),':y','linewidth',3,'markerfacecolor','y');
        plot(SI(1:j,1),SI(1:j,2),'-c','linewidth',3);
        hold off
        frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
        Imlgp = imcrop(frame.cdata, [xd 50 500 500]); % Traj 1 & 2, x-250 for 2
        
        Imlgp = insertText(Imlgp, [7 6], ['Feature conf. ' num2str(k)],'fontsize',35);
        %     imshow(Imlgp);
        
        if record
            writeVideo(writerObj, Imlgp);
        end
        
        j = j + speed;
    end
    
    
    if record
        close(writerObj); % Saves the movie.
    end
    
    k = k + 1;
    
end
