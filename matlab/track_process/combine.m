clear all
clc

% v1 = VideoReader('/home/avishai/Dropbox/transfer/rec_vs_04.avi');
% v2 = VideoReader('/home/avishai/Dropbox/transfer/rec_nn_04.avi');
v1 = VideoReader('/home/avishai/Dropbox/transfer/circ_vs.avi');
v2 = VideoReader('/home/avishai/Dropbox/transfer/circ_nn.avi');

record = 1;
if record
    writerObj = VideoWriter('/home/avishai/Dropbox/transfer/circ.avi'); %my preferred format
    writerObj.FrameRate = 60;
    open(writerObj);
end

while 1
    
    if ~hasFrame(v1) && ~hasFrame(v2)
        break;
    end
    
    if hasFrame(v1)
        I1 = readFrame(v1);
    end
    
    if hasFrame(v2)
        I2 = readFrame(v2);
    end
    
    frame = [I1 I2];
    
    frame = frame(30:end,:,:);
        
    if record
        writeVideo(writerObj, frame);
    end
    
end

if record
    close(writerObj); % Saves the movie.
end