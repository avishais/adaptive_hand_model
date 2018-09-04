clear all
warning('off','all')

data_source = '20';
test_num = 1;
mode = 8;
switch mode
    case 1
        w = [3 3 1 1];
    case 2
        w = [3 3 1 1 1 1 1 1];
    case 3
        w = [3 3 1 1 1 1 1 1 1 1 3 3];
    case 4
        w = [];
    case 5
        w = [60 60 1 1 3 3];
    case 7
        w = [10 10 ones(1,14)];
    case 8
        w = [5 5 3 3 1 1 3 3];%[5 5 3 3 1 1 3 3]; % Last best: [3 3 1 1 1 1 1 1];
end

[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source);

Sr = Xtest;


%% open loop
SRI = zeros(size(Sr,1), 2);
for i = 1:size(Sr,1)
    SRI(i,:) = project2image(Sr(i,1:2), I);
end
    
s = Sr(1,I.state_inx);
S = zeros(size(Sr,1), I.state_dim);
SI = zeros(size(Sr,1), 2);
S(1,:) = s;
SI(1,:) = project2image(s(1:2), I);
loss = 0;
for i = 1:100%size(Sr,1)-1
    a = Sr(i, I.action_inx);
    disp(['Step: ' num2str(i) ', action: ' num2str(a)]);
    [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);
    S(i+1,:) = s;
    loss = loss + norm(s - Sr(i+1, I.state_nxt_inx))^2;
    
    SI(i+1,:) = project2image(s(1:2), I);
    
end
SI = SI(1:i+1,:);

loss = loss / size(Sr,1);

%%

figure(1)
clf
hold on
plot(SRI(:,1),SRI(:,2),'-b','linewidth',3,'markerfacecolor','y');
plot(SI(:,1),SI(:,2),'-r','linewidth',3,'markerfacecolor','y');
axis equal
hold off
axis equal
legend('ground truth','predicted path');
title(['open loop - ' num2str(mode) ', MSE: ' num2str(loss)]);
disp(['Loss: ' num2str(loss)]);

%% Functions

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

