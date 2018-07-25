clear all

rng(0,'twister'); % For reproducibility

f = @(x) 1 + x*5e-2 + sin(x)./x;

n = 1000;
x = linspace(-10,10,n)';
y = f(x) + 0.2*randn(n,1);
y_real_train = f(x);
x_test = 10:0.5:15;
y_real_test = f(x_test);
xt = [-6.5, 5.2, 0];

gprMdl = fitrgp(x,y,'Basis','linear','FitMethod','exact','PredictMethod','exact');

ypred = resubPredict(gprMdl);
y_pred_test = predict(gprMdl, x_test');
yt = predict(gprMdl, xt');

plot(x,y,'b.');
hold on;
plot(x,ypred,'r','LineWidth',1.5);
plot(x,y_real_train,'--r','LineWidth',1.5);
plot(x_test,y_real_test,'--k','LineWidth',1.5);
plot(x_test,y_pred_test,'-k','LineWidth',1.5);
plot(xt, yt, 'ok','markerfacecolor','g');
xlabel('x');
ylabel('y');
legend('Data','GPR predictions');
hold off

%% 
f = @(x, y) x.^2/2 + y.^2/4;

n = 1000;
x = linspace(-10,10,n)';
x = rand(n, 1)*20 - 10;
y = rand(n, 1)*20 - 10;
z_real_train = f(x,y) + 2*randn(n,1);
x_test = rand(100, 1)*30 - 15;
y_test = rand(100, 1)*30 - 15;
z_real_test = f(x_test, y_test) + 2*randn(100,1);
% xt = [-6.5, 5.2, 0];

gprMdl = fitrgp([x,y], z_real_train,'Basis','linear','FitMethod','exact','PredictMethod','exact');

zpred = resubPredict(gprMdl);
z_pred_test = predict(gprMdl, [x_test y_test]);

plot3(x,y,z_real_train,'b.');
hold on;
plot3(x,y,zpred,'or','LineWidth',1.5);
plot3(x_test,y_test,z_real_test,'.k','LineWidth',1.5);
plot3(x_test,y_test,z_pred_test,'ok','LineWidth',1.5);
xlabel('x');
ylabel('y');
legend('Data','GPR predictions');
grid on
hold off

disp(resubLoss(gprMdl));
disp(loss(gprMdl,[x_test y_test],z_real_test));
