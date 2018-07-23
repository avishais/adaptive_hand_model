clear all
clf

num_net = 1;
% Q = load(['../../data/berk_data/data' num2str(num_net) '.mat'], 'Q');
% Q = Q.Q;
% 
% Xtraining = [];
% for i = 1:size(Q,1)
%     Xtraining = [Xtraining; Q{i}.data(:,1:6)];    
% end

% Xtraining = load('../../data/toyData.db');
Xtraining = load('../../data/data_25.db');

% Xtraining = Xtraining(1:300,:);

xmax = max(Xtraining); xmax([1, 5]) = max(xmax([1, 5])); xmax([2, 6]) = max(xmax([2, 6]));
xmin = min(Xtraining); xmin([1, 5]) = min(xmin([1, 5])); xmin([2, 6]) = min(xmin([2, 6]));
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);

% Xtest = Q{end-1}.data(2:end,1:6);
% Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

%% Max motion

r_max = 0;
for i = 1:size(Xtraining,1)
    r = norm(Xtraining(i,1:2)-Xtraining(i,5:6));
    if r > r_max
        r_max = r;
    end        
end

%%

a = action('d');
% x = [0.42 0.43 a];
x = [0.55 0.37 a];
[idx, d] = knnsearch(Xtraining(:,1:4), x(1:4), 'K', 30);
% [idx, d] = rangesearch(Xtraining(:,1:4), x(1:4), 0.03); d = d{1}; idx = idx{1};

data_nn = Xtraining(idx,:);

mind = min(d);
maxd = max(d);

xx = linspace(0,1,100);
yy = xx;
[XX, YY] = meshgrid(xx,yy);

P = zeros(size(XX));

for i = 1:length(xx)
    for j = 1:length(yy)
        P(i,j) = kde([x(1:2) a XX(i,j) YY(i,j)], data_nn);
    end
end
sp = sum(P(1:end));
% P = P/sp;

% [X, Pp] = kdePropagate(x(1:2), a, r_max, Xtraining);
% Pp = Pp/sp;

% Most probable node
[~,i] = max(P(:));
[i_row, i_col] = ind2sub(size(P),i);
x_nxt_prob = [XX(i_row, i_col) YY(i_row, i_col)];

%
figure(1)
clf
hold on
plot3(x(1),x(2),max(zlim),'ok','markerfacecolor','m','markersize',14);
plot3(x_nxt_prob(1),x_nxt_prob(2),max(zlim),'ok','markerfacecolor','c','markersize',14);
surf(XX,YY,P); 
z = zlim;
plot3(Xtraining(:,1),Xtraining(:,2),z(2)*ones(size(Xtraining,1),1),'xk');
plot3(ones(2,1)*x(1), ones(2,1)*x(2), z, ':k', 'linewidth',3);
hold off
axis equal
axis([0 1 0 1]);
view(2)
legend('current state','most probable state');
camroll(-180);
text(0.9, 0.87, z(2), ['action =' what_action(a)],'fontsize',25,'color','white');
text(0.9, 0.92, z(2), ['dist. = ' num2str(mind) ',' num2str(maxd)],'fontsize',25,'color','white');

filen = ['T_' num2str(x(1)) '_' num2str(x(2)) '_' what_action(a) '.png'];
% print(filen, '-dpng', '-r150');


% [0.2211, 0.2513]


%%

% j = 4;
% 
% figure(1)
% clf
% Sr = Xtest(1:j,1:2);
% plot(Sr(:,1),Sr(:,2),'.-b');
% 
% for l = 1:10
%     s = Xtest(1,1:2);
%     S = s;
%     for i = 1:j%size(Xtest,1)
%         a = Xtest(i, 3:4);
%         s = kdePropagate(s, a, r_max, Xtraining);
%         S = [S; s];
%     end
%     
%     hold on
%     plot(S(:,1),S(:,2),'.-k');
%     plot(S(1,1),S(1,2),'or','markerfacecolor','r');
%     hold off
%     drawnow;
% end

function a = action(index)
switch index
    case 'd'
        a = [1 0];
    case 'a'
        a = [0 1];
    case 'w'
        a = [0 0];
    case 'x'
        a = [1 1];
end
end

function str = what_action(a)
if all(a==[0 0])
    str = 'up';
else if all(a==[1 1])
        str = 'down';
    else if all(a==[0 1])
            str = 'left';
        else
            if all(a==[1 0])
                str = 'right';
            end
        end
    end
end
end

