clear all

data_source = '20';

files = dir(fullfile('./cc/', ['cc_' data_source '_*.txt']));
files = struct2cell(files)';
files = sortrows(files, 1);

n = size(files,1);

%%
for i = 148:149
%     f = files{i,1};
    f = findfile(files, i);
    
    D = dlmread(['./cc/' f], ' ');
    disp(['Processing file ' f ' with ' num2str(size(D,1)) ' nodes...']);
      
    flag = 0;
    j = 1;
    while j < size(D,1) && sum(D(j,2:9)==0) <= 6
        j = j + 1;
    end
    
    if j == size(D,1)
        continue;
    end
    
    while sum(D(j,2:9)==0) > 6
        j = j + 1;
    end
    
    D(1:j-1,:) = [];
    
    dlmwrite(['./cc/' f], D, 'delimiter',' ','precision',6);
        
end

%%

function f = findfile(files, inx)

for i = 1:size(files,1)
    str = files{i};
    k = 1; c = 0;
    while str(k)~='.'
        if str(k)=='_'
            c = c + 1;
        end
        if c==2
            break;
        end
        k = k + 1;
    end
    k1 = k + 1;
    while str(k)~='.'
        k = k + 1;
    end
    if str2num(str(k1:k-1))==inx
        f = files{i};
        return;
    end
    
end

end