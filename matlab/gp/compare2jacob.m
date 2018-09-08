clear all
warning('off','all')

data_source = '20';
test_num = 3;
mode = 8;
% w = [1.05 1.05 1 1 2 2 3 3]; % For cyl 25 and mode 8
w = [];
% switch mode
%     case 1
%         w = [3 3 1 1];
%     case 2
%         w = [3 3 1 1 1 1 1 1];
%     case 3
%         w = [3 3 1 1 1 1 1 1 1 1 3 3];
%     case 4
%         w = [];
%     case 5
%         w = [60 60 1 1 3 3];
%     case 7
%         w = [10 10 ones(1,14)];
%     case 8
%         w = [5 5 3 3 1 1 3 3]; % Last best: [5 5 3 3 1 1 3 3];
% end
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source);

%%
dt = 1/15;

DX = zeros(size(Xtraining,1)-2, 4);
for i = 2:size(Xtraining,1)-1
    for j = 1:4
        DX(i-1, j) = (Xtraining(i+1,j) - 2*Xtraining(i,j) + Xtraining(i-1,j))/dt^2;
    end
end

%%
id = randperm(size(DX,1));
id = id(1:1e5);
DX = DX(id,:);

K0 = [100, 100];
% K = fminsearch(@(x)objfunc(x, DX),K0);

K = [-0.995820726947830,1.52591298431857];
%%

px2mm = 0.2621;

DXtest = zeros(size(Xtest,1), 4);
for i = 2:size(Xtest,1)-1
    for j = 1:4
        DXtest(i, j) = (Xtest(i+1,j) - 2*Xtest(i,j) + Xtest(i-1,j))/dt^2;
    end
end

Jt = [1/K(1) 1/K(2); -1/K(1) 1/K(2)]^-1;
i = 59;

mse_J = 0;
mse_GP = 0;
for i = 2:size(Xtest,1)-1
    disp(['Step ' num2str(i)]);
    % Berks prediction
    x = Xtest(i, 1:2);
    x_real_next = Xtest(i, 1:2);
    dq = DXtest(i, 3:4);
    x_next(1) = x(1) + (dq(1)*Jt(1,1) + dq(2)*Jt(1,2))*dt;
    x_next(2) = x(2) + (dq(1)*Jt(2,1) + dq(2)*Jt(2,2))*dt;
    
    [s, s2] = prediction(kdtree, Xtraining, Xtest(i, I.state_inx), Xtest(i, I.action_inx), I, 1);
    
%     figure(1)
%     clf
%     hold on
%     plot(x(1),x(2),'xr');
%     plot(x_next(1), x_next(2), 'og');
%     plot(x_real_next(1), x_real_next(2), 'ob');   
%     plot(s(1), s(2), 'sm');
%     hold off
%     legend('current pos','Jacob','real next','predicted');
%     axis([0 1 0 1]);

    mse_J = mse_J + norm(project2image(x_real_next, I)*px2mm-project2image(x_next, I)*px2mm)^2;
    mse_GP = mse_GP + norm(project2image(x_real_next, I)*px2mm-project2image(s(1:2), I)*px2mm)^2;
    
end
    
mse_J = mse_J/(size(Xtest,1)-2);
mse_GP = mse_GP/(size(Xtest,1)-2);

rmse_J = sqrt(mse_J);
rmse_GP = sqrt(mse_GP);

%%
function f = objfunc(x, DX)

% id = randperm(size(DX,1));
% id = id(1:1e3);

f = 0;
for i = 1:size(DX,1)
    dq = DX(i,3:4);
    dv = DX(i,1:2);
    J = [1/x(1) 1/x(2); -1/x(1) 1/x(2)];
    dvp = inv(J) * dq';
    
    f = f + (dv(1)-dvp(1))^2 + (dv(2)-dvp(2))^2;    
end

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

