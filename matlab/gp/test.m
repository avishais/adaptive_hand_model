clear all

rng(0,'twister'); % For reproducibility

f = @(x) 1 + x*5e-2 + sin(x)./x;

n = 100;
x = linspace(-10,10,n)';
y = f(x) + 0.08*randn(n,1);
y_real_train = f(x);
x_test = 10:0.5:15;
y_real_test = f(x_test);
xt = [-6.5, 5.2, 0];

% gprMdl = fitrgp(x,y,'Basis','linear','FitMethod','exact','PredictMethod','exact');
gprMdl = fitrgp(x,y,'KernelFunction','squaredexponential');

ypred = resubPredict(gprMdl);
[y_pred_test, ~, s2_test] = predict(gprMdl, x_test');
[~, ~, s2] = predict(gprMdl, x);

figure(1)
plot(x,y,'b.');
hold on;
plot(x,ypred,'r','LineWidth',1.5);
plot(x,y_real_train,'--r','LineWidth',1.5);
plot(x_test,y_real_test,'--k','LineWidth',1.5);
plot(x_test,y_pred_test,'-k','LineWidth',1.5);
plot(x,s2(:,1),'g',x,s2(:,2),'g')
plot(x_test,s2_test(:,1),'g',x_test,s2_test(:,2),'g');
% plot(xt, yt, 'ok','markerfacecolor','g');
xlabel('x');
ylabel('y');
legend('Data','GPR predictions');
hold off

% Using gpml

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood
hyp = struct('mean', [], 'cov', [0, 0], 'lik', -1);
hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu_test, s2_test] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test');
[mu, s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x);

figure(2)
plot(x,y,'b.');
hold on;
plot(x_test,y_real_test,'--k','LineWidth',1.5);
plot(x_test,mu_test,'-k','LineWidth',1.5);
plot(x_test,mu_test+s2_test.^2,'g',x_test,mu_test-s2_test.^2,'g')
plot(x,mu+s2.^2,'g',x,mu-s2.^2,'g');
% plot(xt, yt, 'ok','markerfacecolor','g');
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

%%
clear all
load(fullfile(matlabroot,'examples','stats','gprdata2.mat'))

gprMdl1 = fitrgp(x,y,'KernelFunction','squaredexponential');
ypred1 = resubPredict(gprMdl1);

sigma0 = 0.2;
kparams0 = [3.5, 6.2];
gprMdl2 = fitrgp(x,y,'KernelFunction','squaredexponential','KernelParameters',kparams0,'Sigma',sigma0);
ypred2 = resubPredict(gprMdl2);

% Optimize the hyper-parameters
rng default
gprMdl3 = fitrgp(x,y,'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
ypred3 = resubPredict(gprMdl3);

figure(1);
plot(x,y,'r.');
hold on
plot(x,ypred1,'b');
plot(x,ypred2,'g');
plot(x,ypred3,'m');
xlabel('x');
ylabel('y');
legend({'data','default kernel parameters',...
'kparams0 = [3.5,6.2], sigma0 = 0.2','optimized'},...
'Location','Best');
title('Impact of initial kernel parameter values');
hold off
