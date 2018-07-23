function data  = process_data(M)

if 1
    % Clean 0's
    i = 1;
    while (i <= size(M,1))
        if (sum(M(i,2:end)==0) > 7)
            M(i,:) = [];
            continue;
        end
        break;
        i = i + 1;
    end
    
    % Clean when gripping
    i = 1;
    while (i <= size(M,1))
        if all(M(i,6:7)==0)
            M(i,:) = [];
            continue;
        end
        break;
        i = i + 1;
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
data.m1 = (R*data.m1')';
data.m2 = (R*data.m2')';
data.m3 = (R*data.m3')';
data.m4 = (R*data.m4')';
data.obj_pos(:,1:2) = (R*data.obj_pos(:,1:2)')';