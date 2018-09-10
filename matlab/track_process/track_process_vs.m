clear all
clc

px2mm = 0.2621;
M = dlmread('pt_vs_04.txt');

% im_start = 14;
% images_test_folder = '../../data/test_images/pt_vs_04/';
% file_prefix = 'image_test3_';
% file = [images_test_folder, file_prefix, num2str(im_start), '_*.jpg'];
% files = dir(fullfile(file));
% IM = imread([files.folder '/' files.name]);

obj_pos = MovingAvgFilter(M(:,18:19))*px2mm;
carrot_pos = M(:,end-1:end)*px2mm;

carrot_pos(all(carrot_pos==0,2),:) = [];
obj_pos(all(obj_pos==0,2),:) = [];

carrot_pos = [obj_pos(1,:); carrot_pos; obj_pos(1,:)];

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
plot(carrot_pos(:,1)+2,carrot_pos(:,2),'--b','linewidth',3);
plot(obj_pos(:,1)+2,obj_pos(:,2),'-r','linewidth',4);
hold off
axis equal
ylim([42.5 49.5]);
xlim([105 123]);
set(gca, 'fontsize',16);
xlabel('x (mm)','fontsize',22);
ylabel('y (mm)','fontsize',22);
legend({'ref. traj.','actual path'},'location','southwest','fontsize',20);

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
    for j = 2:n
        c = carrot_pos(i-1,:)*(1-lambda(j)) + carrot_pos(i,:)*lambda(j);
        C = [C; c];
    end
end

Ci = [353 1585 1937 size(C,1)];
Pi = [450 1393 2239 size(obj_pos,1)];

figure(2)
clf 
hold on
plot(C(:,1),C(:,2),'.-b');
   
MSE = 0;
max_err = 0;
for i = 1:size(obj_pos,1)
    if i < Pi(1)
        idx = knnsearch(C(1:Ci(1),:), obj_pos(i,:));
    end
    if i > Pi(1) && i < Pi(2)
        idx = knnsearch(C(Ci(1)+1:Ci(2),:), obj_pos(i,:));
        idx = idx + Ci(1);
    end
    if i > Pi(2) && i < Pi(3)
        idx = knnsearch(C(Ci(2)+1:Ci(3),:), obj_pos(i,:));
        idx = idx + Ci(2);
    end
    if i > Pi(3) && i < Pi(4)
        idx = knnsearch(C(Ci(3)+1:Ci(4),:), obj_pos(i,:));
        idx = idx + Ci(3);
    end
    
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
record = 0;

files = dir(fullfile(images_test_folder, '*.jpg'));
files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);

if record
    writerObj = VideoWriter('/home/avishai/Dropbox/transfer/rec_vs_04.avi'); %my preferred format
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
        frame.cdata = insertText(frame.cdata,[260 8],'visual servoing','fontsize',18);
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