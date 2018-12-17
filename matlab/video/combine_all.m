clear all

test_num = 1;

%% Combine

v1 = VideoReader('traj_26_1_all_dm.avi');
v2 = VideoReader('traj_30_3_all_dm.avi');
v3 = VideoReader('traj_36_2_all_dm.avi');

record = 1;
if record
    writerObj = VideoWriter(['trajs_all_' num2str(test_num) '.avi']); %my preferred format
    writerObj.FrameRate = 60;
    open(writerObj);
end

j = 1;
while 1
    
    if ~hasFrame(v1) && ~hasFrame(v2) && ~hasFrame(v3) 
        for j = 1:250
            writeVideo(writerObj, frame);
        end
        break;
    end
    
    if hasFrame(v1)
        I1 = readFrame(v1);
    end
    
    if hasFrame(v2)
        I2 = readFrame(v2);
    end
    
    if j==1 || j > 100
        if hasFrame(v3)
            I3 = readFrame(v3);
        end
    end
    j = j + 1;
    
    
    frame = [zeros(size(I1)) I1 I2 I3];
    
    frame = insertText(frame, [30 10], ['Action sequence ' num2str(test_num)],'fontsize',35, 'BoxOpacity',0.8,'TextColor','white','BoxColor','blac');
     
    frame = insertText(frame, [30 80], 'Yellow - ref. traj.','fontsize',35, 'BoxOpacity',0.8,'TextColor','Yellow','BoxColor','black');
    frame = insertText(frame, [30 140], 'Cyan - predicted traj.','fontsize',35, 'BoxOpacity',0.8,'TextColor','cyan','BoxColor','black');
    
%     imshow(frame);
    
    if record
        writeVideo(writerObj, frame);
    end
    
end

if record
    close(writerObj); % Saves the movie.
end