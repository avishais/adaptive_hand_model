clear all
clc

px2mm = 0.2621;

M = dlmread('pt_vs_circ.txt');
im_start = 19;
images_test_folder = '../../data/test_images/pt_vs_circ/';

file_prefix = 'image_test3_';
file = [images_test_folder, file_prefix, num2str(im_start), '*.jpg'];
files = dir(fullfile(file));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);
f = find_file(file_prefix, im_start, files);
IM = imread([images_test_folder f]);

obj_pos = (M(:,18:19))*px2mm;
carrot_pos = M(:,end-1:end)*px2mm;

carrot_pos(all(carrot_pos==0,2),:) = [];
obj_pos(all(obj_pos==0,2),:) = [];

carrot_pos = [carrot_pos; carrot_pos(1,:)];

i = 2;
while i < size(carrot_pos,1)
    if all(carrot_pos(i,:)==carrot_pos(i-1,:))
        carrot_pos(i,:)=[];
    else
        i = i + 1;
    end
end

figure(1)
clf
% imshow(IM);
hold on
plot(carrot_pos(:,1),carrot_pos(:,2),'--b','linewidth',3);
plot(obj_pos(:,1),obj_pos(:,2),'-r','linewidth',4);
hold off
axis equal
% ylim([42.5 49.5]);
% xlim([105 123]);
set(gca, 'fontsize',12);
xlabel('x (mm)','fontsize',17);
ylabel('y (mm)','fontsize',17);
legend({'ref. traj.','actual path'},'location','southwest','fontsize',12);

% print(['cl_vs.png'],'-dpng','-r150');

%% RMSE compare

L = 0;
for i = 2:size(carrot_pos,1)
    L = L + norm(carrot_pos(i,:)-carrot_pos(i-1,:));
end
dd = L / size(obj_pos,1);

C = carrot_pos(1,:);
for i = 2:size(carrot_pos,1)
    d = norm(carrot_pos(i,:)-carrot_pos(i-1,:));
    n = ceil(d/dd);
    lambda = linspace(0,1,n);
    for j = 1:n
        c = carrot_pos(i-1,:)*(1-lambda(j)) + carrot_pos(i,:)*lambda(j);
        C = [C; c];
    end
end

figure(2)
clf 
hold on
plot(C(:,1),C(:,2),'.-b');
   
MSE = 0;
max_err = 0;
for i = 1:size(obj_pos,1)
    idx = knnsearch(C, obj_pos(i,:));
    err = norm(C(idx,:)-obj_pos(i,:));
    MSE = MSE + err^2;  
    plot(obj_pos(i,1),obj_pos(i,2),'or');
    plot([obj_pos(i,1) C(idx,1)], [obj_pos(i,2) C(idx,2)],'-k');
    
    if err > max_err
        max_err = err;
    end
end
MSE = MSE/size(obj_pos,1);

disp(['RMSE ' num2str(sqrt(MSE))]);
disp(['Max error: ' num2str(max_err)]);

hold off
axis equal

%%
record = 1;

files = dir(fullfile(images_test_folder, '*.jpg'));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);

if record
    writerObj = VideoWriter('/home/avishai/Dropbox/transfer/circ_vs.avi'); %my preferred format
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
        frame.cdata = imcrop(frame.cdata, [380 1 300 300]);
        frame.cdata = insertText(frame.cdata,[10 36],'visual servoing','fontsize',18);
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
    windowSize = 13;
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