clear all

num_net = 1;
Q = load(['../../data/data' num2str(num_net) '.mat'], 'Q');
Q = Q.Q;

Xtraining = [];
for i = 1:size(Q,1)-1
    Xtraining = [Xtraining; Q{i}.data(:,1:6)];    
end
xmax = max(Xtraining); xmax([1, 5]) = max(xmax([1, 5])); xmax([2, 6]) = max(xmax([2, 6]));
xmin = min(Xtraining); xmin([1, 5]) = min(xmin([1, 5])); xmin([2, 6]) = min(xmin([2, 6]));
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);

Xtest = Q{end-1}.data(2:end,1:6);
Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

% Xtraining(:,3:4) = Xtraining(:,3:4)*5;
% Xtest(:,3:4) = Xtest(:,3:4)*5;

%% Max motion

r_max = 0;
for i = 1:size(Xtraining,1)
    r = norm(Xtraining(1:2)-Xtraining(5:6));
    if r > r_max
        r_max = r;
    end        
end

%%

a = [1 1];
x = [0.5 0.5 a];
[idx, d] = knnsearch(Xtraining(:,1:4), x(1:4), 'K', 1000);
mind = min(d);
maxd = max(d);
data_nn = Xtraining(idx,:);
% idx = rangesearch(Xtraining(:,1:4), x(1:4), 0.1);
% data_nn = Xtraining(idx{1},:);

xx = linspace(0,1,25);
yy = xx;
[XX, YY] = meshgrid(xx,yy);

P = zeros(size(XX));

for i = 1:length(xx)
    for j = 1:length(yy)
        P(i,j) = kde([x(1:2) a XX(i,j) YY(i,j)], data_nn);
    end
end
sp = sum(P(1:end));
P = P/sp;

[X, Pp] = kdePropagate(x(1:2), a, r_max, Xtraining);
Pp = Pp/sp;

%
surf(XX,YY,P); 
hold on
plot3(ones(2,1)*x(1), ones(2,1)*x(2), zlim, ':k', 'linewidth',3);
plot3(x(1),x(2),max(zlim),'ok','markerfacecolor','m','markersize',16);
% plot3(X(:,1),X(:,2),Pp,'ok','markerfacecolor','y');
text(0.05, 0.95, action(a),'fontsize',25,'color','white');
text(0.05, 0.9, ['dist. = ' num2str(mind) ',' num2str(maxd)],'fontsize',25,'color','white');
hold off
axis equal
view(2)



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

function str = action(a)
if all(a==[0 0])
    str = 'action = down';
else if all(a==[1 1])
        str = 'action = up';
    else if all(a==[1 0])
            str = 'action = left';
        else
            if all(a==[0 1])
                str = 'action = right';
            end
        end
    end
end
end

