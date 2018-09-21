function data = process_data(M)

if 1
    
    flag = 0;
    j = 1;
    while j < size(M,1) && sum(M(j,2:9)==0) <= 6
        j = j + 1;
    end
    
    if j < size(M,1) 
        while abs(sum(M(j,6:7))) < 1e-3 
            j = j + 1;
        end
    end
    
    j_new = j;
    
    while all(M(j_new,6:7)==[0.2,0.2])
        j_new = j_new + 1;        
    end
    
    if all(M(j_new,6:7)==[0,0])
        j = j_new;
        while all(M(j,6:7)==[0,0])
            j = j + 1;
        end
    end
    
    M(1:j-1,:) = [];
    
    if all(M(end,2:5)==0)
        j = size(M,1);
        while all(M(j,2:5)==0)
            j = j - 1;
        end
        M(j+1:end,:) = [];        
        
        data.fail_type = 'overload';
    else
        % Remove constant object position in the end (after drop)
        j = size(M,1);
        while all(M(j,18:19)==M(j-1,18:19))
            j = j - 1;
        end
        M(j:end,:) = [];
        
        % Check jump in data due to fall
        j = size(M,1);
        while j > size(M,1)-60
            if any(abs(M(j,18:19)-M(j-1,18:19)) >= 4 )
                while any(abs(M(j,18:19)-M(j-1,18:19)) >= 4 )
                    j = j - 1;
                end
                
                M(j+1:end,:) = [];
                break;
            end
            j = j - 1;
        end
        
        data.fail_type = 'drop';
    end
end

% Proccess
data.T = M(:,1);
data.T = data.T-data.T(1);
data.dt = mean(diff(data.T));

data.n = length(data.T);
data.base_pos = mean(M(:,21:23)); 
data.theta = pi - data.base_pos(3);

data.act_pos = M(:,2:3);
data.act_load = M(:,4:5);
data.ref_vel = M(:,6:7);
data.ref_pos = M(:,8:9);
data.m1 = M(:,10:11) - repmat(data.base_pos(1:2), data.n, 1); 
data.m2 = M(:,12:13) - repmat(data.base_pos(1:2), data.n, 1);  
data.m3 = M(:,14:15) - repmat(data.base_pos(1:2), data.n, 1);  
data.m4 = M(:,16:17) - repmat(data.base_pos(1:2), data.n, 1);  
data.obj_pos = M(:,18:20) - repmat([data.base_pos(1:2) data.theta], data.n, 1); 

data.new_base_pos(1:2) = [0 0];
data.new_base_pos(3) = data.base_pos(3) + data.theta;

R = [cos(data.theta) -sin(data.theta); sin(data.theta) cos(data.theta)];
data.m1 = MovingAvgFilter((R*data.m1')');
data.m2 = MovingAvgFilter((R*data.m2')');
data.m3 = MovingAvgFilter((R*data.m3')');
data.m4 = MovingAvgFilter((R*data.m4')');
% data.obj_pos(:,1:2) = MovingAvgFilter((R*data.obj_pos(:,1:2)')', 20);
data.obj_pos(:,1:2) = MovingAvgFilter((R*data.obj_pos(:,1:2)')');

end

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
