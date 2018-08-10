clear all

warning('off','all')

test_num = 3;
folder_prefix = ['ca_25_test' num2str(test_num)];
images_test_folder = ['../../data/test_images/' folder_prefix '/'];
file_prefix = ['image_test' num2str(test_num) '_'];

files = dir(fullfile(images_test_folder, '*.jpg'));

mode = 5;
[Xtraining, Xtest, kdtree, I] = load_data(mode, test_num);

if test_num==2
    ix = 386;
end
if test_num==3
    ix = 381;
end

s = Xtest(1,I.state_inx);
sp = s;

Sd = [];
Sp = [];
speed = 1;
for i = 1:speed:size(Xtest,1)
    sd = project2image(Xtest(i,1:2), I);
    Sd = [Sd; sd];
    
    filename = find_file(file_prefix, ix, files);
    IM = imread([images_test_folder filename]);
    
    a = Xtest(i, I.action_inx);
    
    spd = project2image(sp(1:2), I);
    Sp = [Sp; spd];
    [sp, s2] = prediction(kdtree, Xtraining, sp, a, I, 1);
       
    ix = ix + speed;
end

    figure(1)
    clf
    imshow(IM);
    hold on
    plot(Sd(:,1),Sd(:,2),'y','linewidth',4);
    plot(Sp(:,1),Sp(:,2),'c','linewidth',4);
    plot(sd(1),sd(2),'ok','markerfacecolor','y','markersize',10);
    plot(spd(1),spd(2),'ok','markerfacecolor','c','markersize',10);
    hold off
    
%     drawnow;
% end



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