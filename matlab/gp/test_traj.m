clear all

record = 1;

warning('off','all')

test_num = 3;
folder_prefix = ['ca_25_test' num2str(test_num)];
images_test_folder = ['../../data/test_images/' folder_prefix '/'];
file_prefix = ['image_test' num2str(test_num) '_'];

files = dir(fullfile(images_test_folder, '*.jpg'));

mode = 5;
w = 1;
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num);

%%
s = Xtest(1,I.state_inx);
sp = s;
Sp = sp;
for i = 1:size(Xtest,1)
    disp(['Step: ' num2str(i)]);
    a = Xtest(i, I.action_inx);
    [sp, s2] = prediction(kdtree, Xtraining, sp, a, I, 1);
    Sp = [Sp; sp];
end


%%
if record
    writerObj = VideoWriter(['/home/avishai/Dropbox/transfer/test_traj_' num2str(test_num) '_' num2str(mode) '.avi']); %my preferred format
    writerObj.FrameRate = 60;
    open(writerObj);
end

if test_num==2
    ix = 386;
end
if test_num==3
%     ix = 381;
    ix = 1419;
end

Sd = [];
Spd = [];
speed = 1;
for i = 1:speed:size(Sp,1)-1
    sd = project2image(Xtest(i,1:2), I);
    Sd = [Sd; sd];
    
    filename = find_file(file_prefix, ix, files);
    IM = imread([images_test_folder filename]);
       
    spd = project2image(Sp(i,1:2), I);
    Spd = [Spd; spd];
       
    ix = ix + speed;
% end

    figure(1)
    clf
    imshow(IM);
    hold on
    plot(Sd(:,1),Sd(:,2),'y','linewidth',4);
    plot(Spd(:,1),Spd(:,2),':c','linewidth',4);
    plot(sd(1),sd(2),'ok','markerfacecolor','y','markersize',10);
    plot(spd(1),spd(2),'pk','markerfacecolor','c','markersize',10);
    hold off
    legend('Actual','Predicted');
    
    drawnow;
    if record
        frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
        frame.cdata = imcrop(frame.cdata, [290 1 880-290 602]);
        writeVideo(writerObj, frame);
    end
end

if record
    close(writerObj); % Saves the movie.
end



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




function filename = find_file(file_prefix, index, files)

f = [file_prefix num2str(index)];

for i = 1:size(files,1)
    
    str = files(i).name;
    j = 1;
    while j <= length(f) && f(j)==str(j)
        j = j + 1;
    end
    if str(j)=='_'
        ix = i;
        break;
    end
end
filename = files(ix).name;
    
end