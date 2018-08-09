clear all

test_num = 3;
folder_prefix = ['ca_25_test' num2str(test_num)];
images_test_folder = ['../../data/test_images/' folder_prefix '/'];
file_prefix = ['image_test' num2str(test_num) '_'];

files = dir(fullfile(images_test_folder, '*.jpg'));

mode = 1;
[Xtraining, Xtest, kdtree, I] = load_data(mode, test_num);

if test_num==2
    ix = 386;
end
if test_num==3
    ix = 381;
end

Sd = [];
speed = 5;
for i = 1:speed:size(Xtest,1)
    s = Xtest(i,1:2);
    sd = project2image(s, I);
    Sd = [Sd; sd];
    
    filename = find_file(file_prefix, ix, files);
    IM = imread([images_test_folder filename]);
    
%     disp([num2str(ix), filename])
    
    figure(1)
    clf
    imshow(IM);
    hold on
    plot(Sd(:,1),Sd(:,2),'y','linewidth',4);
    plot(sd(1),sd(2),'ok','markerfacecolor','y','markersize',10);
    hold off
    
    drawnow;
    ix = ix + speed;
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