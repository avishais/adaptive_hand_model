clear all

X = load('data.txt');

Xtrain = X(:,1:3);
Xtest = X(:,4:6);
sigma = sqrt(X(:,7));

Xtrain(41:end,:) = [];

%%

figure(1)
clf
hold on
plot3(Xtrain(:,1),Xtrain(:,2),Xtrain(:,3),'ok');
plot3(Xtest(:,1),Xtest(:,2),Xtest(:,3),'or');
for i = 1:size(sigma,1)
    plot3(Xtest(i,1)*[1 1],Xtest(i,2)*[1 1],[Xtest(i,3)+sigma(i) Xtest(i,3)-sigma(i)],'-m');
end
hold off
grid on


%%

% % gprMdl = fitrgp(Xtrain(:,1),Xtrain(:,2),'Basis','linear','FitMethod','exact','PredictMethod','exact');
% gprMdl = fitrgp(Xtrain(:,1),Xtrain(:,2),'KernelFunction','squaredexponential','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus'));
% % gprMdl = fitrgp(Xtrain(:,1),Xtrain(:,2),'Basis','constant','FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential','KernelParameters',[1.0520;0.0375],'Sigma',0.2,'Standardize',1);
% 
% [mu_ml, sigma_ml] = predict(gprMdl, Xtest(:,1));
% 
% figure(2)
% clf
% hold on
% fill([Xtest(:,1); flipud(Xtest(:,1))],[mu_ml+sigma_ml; flipud(mu_ml-sigma_ml)],[1 1 0.9]);
% plot(Xtrain(:,1),Xtrain(:,2),'ok');
% plot(Xtest(:,1),mu_ml,'.-b');
% hold off
% ylim([-2 2]);
% 
% figure(1)
% hold on
% plot(Xtest(:,1),mu_ml,'.-b');
% hold off