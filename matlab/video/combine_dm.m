clear all

test_num = 1;

%% Combine

v1 = VideoReader(['traj' num2str(1) '.avi']);
v2 = VideoReader(['traj' num2str(1) '_dm.avi']);
v3 = VideoReader(['traj' num2str(2) '.avi']);
v4 = VideoReader(['traj' num2str(2) '_dm.avi']);
v5 = VideoReader(['traj' num2str(3) '.avi']);
v6 = VideoReader(['traj' num2str(3) '_dm.avi']);

record = 1;
if record
    writerObj = VideoWriter(['trajs.avi']); %my preferred format
    writerObj.FrameRate = 60;
    open(writerObj);
end

while 1
    
    if ~hasFrame(v1) && ~hasFrame(v2) && ~hasFrame(v3) && ~hasFrame(v4) && ~hasFrame(v5) && ~hasFrame(v6)
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
    
    frame = [I1 I2; I3 I4; I5 I6];
    frame = [frame zeros(size(frame,1), 500, 3)];
    
    frame = insertText(frame, [350 410], 'Test trajectory 1','fontsize',35, 'BoxOpacity',0.8,'TextColor','black','BoxColor','white');
    frame = insertText(frame, [350 410+500], 'Test trajectory 2','fontsize',35, 'BoxOpacity',0.8,'TextColor','black','BoxColor','white');
    frame = insertText(frame, [350 410+1000], 'Test trajectory 3','fontsize',35, 'BoxOpacity',0.8,'TextColor','black','BoxColor','white');
    
    frame = insertText(frame, [1020 650], 'Yellow - ref. traj.','fontsize',35, 'BoxOpacity',0.8,'TextColor','Yellow','BoxColor','black');
    frame = insertText(frame, [1020 700], 'Cyan - predicted traj.','fontsize',35, 'BoxOpacity',0.8,'TextColor','cyan','BoxColor','black');
    
    imshow(frame);
    
    if record
        writeVideo(writerObj, frame);
    end
    
end

if record
    close(writerObj); % Saves the movie.
end