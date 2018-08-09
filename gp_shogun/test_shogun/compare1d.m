clear all

X = load('data.txt');

Xtrain = X(:,1:2);
Xtest = X(:,3:4);
sigma = sqrt(X(:,5));

Xtrain(5:end,:) = [];

%%

figure(1)
clf
hold on
fill([Xtest(:,1); flipud(Xtest(:,1))],[Xtest(:,2)+sigma; flipud(Xtest(:,2)-sigma)],[1 1 0.9]);
plot(Xtrain(:,1),Xtrain(:,2),'ok');
plot(Xtest(:,1),Xtest(:,2),'.-r');
hold off
ylim([-2 2]);


%%

% gprMdl = fitrgp(Xtrain(:,1),Xtrain(:,2),'Basis','linear','FitMethod','exact','PredictMethod','exact');
gprMdl = fitrgp(Xtrain(:,1),Xtrain(:,2),'KernelFunction','squaredexponential','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus'));
% gprMdl = fitrgp(Xtrain(:,1),Xtrain(:,2),'Basis','constant','FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential','KernelParameters',[1.0520;0.0375],'Sigma',0.2,'Standardize',1);

[mu_ml, sigma_ml] = predict(gprMdl, Xtest(:,1));

figure(2)
clf
hold on
fill([Xtest(:,1); flipud(Xtest(:,1))],[mu_ml+sigma_ml; flipud(mu_ml-sigma_ml)],[1 1 0.9]);
plot(Xtrain(:,1),Xtrain(:,2),'ok');
plot(Xtest(:,1),mu_ml,'.-b');
hold off
ylim([-2 2]);

figure(1)
hold on
plot(Xtest(:,1),mu_ml,'.-b');
hold off