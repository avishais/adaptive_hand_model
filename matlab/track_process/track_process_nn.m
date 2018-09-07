clear all
clc

M = dlmread('pt_nn_04.txt');

im_start = 4292;
images_test_folder = '../../data/test_images/pt_nn_04/';
file_prefix = 'image_test3_';
file = [images_test_folder, file_prefix, num2str(im_start), '*.jpg'];
files = dir(fullfile(file));
IM = imread([files.folder '/' files.name]);

obj_pos = MovingAvgFilter(M(:,18:19));
carrot_pos = M(:,end-1:end);

carrot_pos(all(carrot_pos==0,2),:) = [];
carrot_pos(all(carrot_pos==-1,2),:) = [];
obj_pos(all(obj_pos==0,2),:) = [];

carrot_pos = [obj_pos(1,:); carrot_pos; obj_pos(1,:)];
obj_pos(end-10:end,:) = [];
obj_pos = [obj_pos; 
    421.2 171.4
    420.9 171.3
    419.900000000000,170.713999906413
    418.800000000000,171.293518346954
    417.700000000000,170.726910687672
    416.600000000000,170.821454800080
    414.600000000000,171.264068246684
    obj_pos(1,:)];

figure(1)
clf
imshow(IM);
hold on
plot(carrot_pos(:,1),carrot_pos(:,2),'--b');
plot(obj_pos(:,1),obj_pos(:,2),'-r');
hold off
axis equal

%%
record = 1;

files = dir(fullfile(images_test_folder, '*.jpg'));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);

if record
    writerObj = VideoWriter('/home/avishai/Dropbox/transfer/rec_nn_04.avi'); %my preferred format
    writerObj.FrameRate = 60;
    open(writerObj);
end

ix = im_start;
speed = 5;
for i = 1:speed:size(obj_pos,1)
    
    filename = find_file(file_prefix, ix, files);
    
    IM = imread([images_test_folder filename]);
    
    figure(1)
    clf
    imshow(IM);
    hold on
    plot(carrot_pos(:,1),carrot_pos(:,2),'-y','linewidth',2);
    plot(obj_pos(1:i,1),obj_pos(1:i,2),'-r','linewidth',2);
    hold off
    
    ix = ix + speed; 
    
    if record
        frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
        frame.cdata = imcrop(frame.cdata, [360 90 770-360 420-90]);
        frame.cdata = insertText(frame.cdata,[220 8],'with learned model','fontsize',18);
%         imshow(frame.cdata);
        writeVideo(writerObj, frame);
    end
%     drawnow;
end

if record
    close(writerObj); % Saves the movie.
end

%%
function y = MovingAvgFilter(x, windowSize)

if nargin==1
    windowSize = 20;
end

y = x;

w = floor(windowSize/2);
for j = 1:size(x,2)
    for i = w+1:size(x,1)-w-1
        
        y(i,j) = sum(x(i-w:i+w,j))/length(i-w:i+w);
        
    end
    
end
end

function filename = find_file(file_prefix, index, files)

f = [file_prefix num2str(index)];

for i = 1:size(files,1)
    
    str = files(i);
    str = str{1};
    j = 1;
    while j <= length(f) && f(j)==str(j)
        j = j + 1;
    end
    if str(j)=='_'
        ix = i;
        break;
    end
end
filename = files(ix);
filename = filename{1};
    
end

