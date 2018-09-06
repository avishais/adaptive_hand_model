clear all

data_source = '20';

files = dir(fullfile('./cc/', ['cc_' data_source '_*.txt']));
files = struct2cell(files)';
files = sortrows(files, 1);

n = size(files,1);

%%
for i = 50:70
    f = files{i,1};
    
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