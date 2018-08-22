clear all

addpath('../');

images_test_folder = '../../../data/misc/for_paper_01/';
file_prefix = 'image_test3_';

is = 966;

% x motion
% id = 1037;
% ix = 1699;

% y motion
id = 1705;
ix = 2655;


files = dir(fullfile(images_test_folder, '*.jpg'));

filename = find_file(file_prefix, is+ix-1, files);

IM = imread([images_test_folder filename]);

imshow(IM);

%%

X = dlmread('../../../data/misc/for_paper_01.txt');

O = X(:,18:19);
A = X(:,6:7);

hold on
plot(O(id:ix,1),O(id:ix,2),'-y','linewidth',7);
plot(O(ix,1),O(ix,2),'ok','markerfacecolor','y','markersize',14);
hold off

frame = getframe(gcf); 
IM = frame.cdata;
IM = imcrop(IM,[582 160 1339-582 809-160]);

imshow(IM);

% imwrite(IM, 'ymotion.png');



%%

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