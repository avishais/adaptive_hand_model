clear all

X = load('path.txt');

mode = 5;
switch mode
    case 1
        action_inx = 3:4;
        state_inx = 1:2;
        state_nxt_inx = 5:6;
    case 2
        action_inx = 7:8;
        state_inx = 1:6;
        state_nxt_inx = 9:14;
    case 3
        action_inx = 11:12;
        state_inx = 1:10;
        state_nxt_inx = 13:22;
    case 4
        action_inx = 5:6;
        state_inx = 1:4;
        state_nxt_inx = 7:10;
    case 5
        action_inx = 5:6;
        state_inx = 1:4;
        state_nxt_inx = 7:10;
    case 6
        action_inx = 13:14;
        state_inx = 1:12;
        state_nxt_inx = 15:26;
    case 7
        action_inx = 15:16;
        state_inx = 1:14;
        state_nxt_inx = 17:30;
    case 8
        action_inx = 7:8;
        state_inx = 1:6;
        state_nxt_inx = 9:14;
end

Sr = X(:, state_inx);
S = X(:, state_nxt_inx);
A = X(:, action_inx);


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