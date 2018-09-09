clear all

test_num = 3;

%% Combine

v1 = VideoReader(['traj_1_' num2str(test_num) '_dm.avi']);
v2 = VideoReader(['traj_2_' num2str(test_num) '_dm.avi']);
v3 = VideoReader(['traj_3_' num2str(test_num) '_dm.avi']);
v4 = VideoReader(['traj_4_' num2str(test_num) '_dm.avi']);
v5 = VideoReader(['traj_5_' num2str(test_num) '_dm.avi']);
v6 = VideoReader(['traj_6_' num2str(test_num) '_dm.avi']);
v7 = VideoReader(['traj_7_' num2str(test_num) '_dm.avi']);


record = 1;
if record
    writerObj = VideoWriter(['trajs_fc_' num2str(test_num) '.avi']); %my preferred format
    writerObj.FrameRate = 60;
    open(writerObj);
end

while 1
    
    if ~hasFrame(v1) && ~hasFrame(v2) && ~hasFrame(v3) && ~hasFrame(v4) && ~hasFrame(v5) && ~hasFrame(v6) && ~hasFrame(v7)
        break;
    end
    
    if hasFrame(v1)
        I1 = readFrame(v1);
    end
    
    if hasFrame(v2)
        I2 = readFrame(v2);
    end
    
    if hasFrame(v3)
        I3 = readFrame(v3);
    end
    
    if hasFrame(v4)
        I4 = readFrame(v4);
    end
    
    if hasFrame(v5)
        I5 = readFrame(v5);
    end
    
    if hasFrame(v6)
        I6 = readFrame(v6);
    end
    
    if hasFrame(v7)
        I7 = readFrame(v7);
    end
    
    frame = [zeros(size(I1)) I1 I2 I3; I4 I5 I6 I7];
    
    frame = insertText(frame, [30 10], ['Test trajectory ' num2str(test_num)],'fontsize',35, 'BoxOpacity',0.8,'TextColor','white','BoxColor','blac');
     
    frame = insertText(frame, [30 80], 'Yellow - ref. traj.','fontsize',35, 'BoxOpacity',0.8,'TextColor','Yellow','BoxColor','black');
    frame = insertText(frame, [30 140], 'Cyan - predicted traj.','fontsize',35, 'BoxOpacity',0.8,'TextColor','cyan','BoxColor','black');
    
    imshow(frame);
    
    if record
        writeVideo(writerObj, frame);
    end
    
end

if record
    close(writerObj); % Saves the movie.
end