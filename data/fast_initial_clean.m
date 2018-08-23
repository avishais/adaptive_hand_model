clear all

data_source = '20';

files = dir(fullfile('./ca/', ['ca_' data_source '_*.txt']));
files = struct2cell(files)';
files = sortrows(files, 1);

n = size(files,1);

%%
for i = 31:n
    f = files{i,1};
    
    D = dlmread(['./ca/' f], ' ');
    
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
    
    dlmwrite(['./ca/' f], D, 'delimiter',' ','precision',6);
        
end