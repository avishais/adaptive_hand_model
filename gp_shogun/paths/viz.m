clear all

X = load('path.txt');

Sr = X(:,1:2);
S = X(:,5:6);
A = X(:,3:4);


%%
figure(2)
clf
plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
hold on
plot(S(:,1),S(:,2),'.-r');
plot(S(1,1),S(1,2),'or','markerfacecolor','r');
hold off
axis equal
legend('ground truth','predicted path');
% title(['open loop - ' num2str(mode) ', MSE: ' num2str(loss)]);
% disp(['Loss: ' num2str(loss)]);