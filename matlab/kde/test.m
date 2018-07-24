clear all
clf

mode = 2;
file = ['../../data/data_25_' num2str(mode)];

Q = load([file '.mat'], 'Q');
Q = Q.Q;
% 
% Xtraining = [];
% for i = 1:size(Q,1)
%     Xtraining = [Xtraining; Q{i}.data(:,1:6)];    
% end

% Xtraining = load('../../data/toyData.db');
Xtraining = load([file '.db']);

% Xtraining = Xtraining(1:300,:);

xmax = max(Xtraining); xmax([1, 5]) = max(xmax([1, 5])); xmax([2, 6]) = max(xmax([2, 6]));
xmin = min(Xtraining); xmin([1, 5]) = min(xmin([1, 5])); xmin([2, 6]) = min(xmin([2, 6]));
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);

Xtest = Q{20}.data(2:end-1,:);
Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

I.action_inx = Q{1}.action_inx;
I.state_inx = Q{1}.state_inx;
I.state_nxt_inx = Q{1}.state_nxt_inx;
I.state_dim = length(I.state_inx);

clear Q
%% Max motion

r_max = 0;
no = zeros(size(Xtraining,1),1);
for i = 1:size(Xtraining,1)
    r = norm(Xtraining(i,I.state_inx)-Xtraining(i,I.state_nxt_inx));
    no(i,:) = r;
    if r > r_max
        r_max = r;
    end        
end

if mode == 2
    r_max = mean(no);
else
    r_max = 0.0015*mean(no);
end

clear r no
%%

if I.state_dim==2
    tc = 20;
    a = Xtraining(tc, I.action_inx); % action('d');
    x = Xtraining(tc, I.state_inx);
    x_next = Xtraining(tc,I.state_nxt_inx);
    
    % [idx, d] = knnsearch(Xtraining(:,[state_inx action_inx]), [x a], 'K', 100);
    [idx, d] = rangesearch(Xtraining(:,[I.state_inx I.action_inx]), [x a], 0.01); d = d{1}; idx = idx{1};
    
    data_nn = Xtraining(idx,:);
    
    mind = min(d);
    maxd = max(d);
    
    xx = linspace(0,1,100);
    yy = xx;
    [XX, YY] = meshgrid(xx,yy);
    
    P = zeros(size(XX));
    
    for i = 1:length(xx)
        for j = 1:length(yy)
            P(i,j) = kde([x a XX(i,j) YY(i,j)], data_nn, I);
        end
    end
    sp = sum(P(1:end));
    % P = P/sp;
    
    % [X, Pp] = kdePropagate(x, a, r_max, Xtraining, I);
    % Pp = Pp/sp;
    
    % Most probable node
    [~,i] = max(P(:));
    [i_row, i_col] = ind2sub(size(P),i);
    x_nxt_prob = [XX(i_row, i_col) YY(i_row, i_col)];
    
    %
    figure(1)
    clf
    subplot(121)
    hold on
    plot3(x(1),x(2),max(zlim),'ok','markerfacecolor','m','markersize',12);
    plot3(x_nxt_prob(1),x_nxt_prob(2),max(zlim),'ok','markerfacecolor','c','markersize',12);
    plot3(x_next(1),x_next(2),max(zlim),'ok','markerfacecolor','g','markersize',12);
    surf(XX,YY,P);
    z = zlim;
    % plot3(Xtraining(:,1),Xtraining(:,2),z(2)*ones(size(Xtraining,1),1),'xk');
    % plot3(X(:,1),X(:,2),Pp,'ok','markerfacecolor','y','markersize',8);
    plot3(ones(2,1)*x(1), ones(2,1)*x(2), z, ':k', 'linewidth',3);
    hold off
    axis equal
    axis([0 1 0 1]);
    view(2)
    legend('current state','most probable state','real motion');
    camroll(-180);
    text(0.9, 0.87, z(2), ['action =' what_action(a)],'fontsize',25,'color','white');
    text(0.9, 0.92, z(2), ['dist. = ' num2str(mind) ',' num2str(maxd)],'fontsize',25,'color','white');
    
    subplot(122)
    hold on
    plot3(x(1),x(2),max(zlim),'ok','markerfacecolor','m','markersize',8);
    plot3(x_nxt_prob(1),x_nxt_prob(2),max(zlim),'ok','markerfacecolor','c','markersize',8);
    plot3(x_next(1),x_next(2),max(zlim),'ok','markerfacecolor','g','markersize',8);
    hold off
    grid on
    axis equal
    w = 1.4*max([norm(x-x_next), norm(x-x_nxt_prob), norm(x_next-x_nxt_prob)]);
    axis([x(1)-w x(1)+w x(2)-w x(2)+w]);
    camroll(-180);
    legend('current state','most probable state','real motion');
    
    filen = ['T_' num2str(x(1)) '_' num2str(x(2)) '_' what_action(a) '.png'];
    % print(filen, '-dpng', '-r150');
    
    clear xx yy w
end
%% 

j_min = 38; j_max = 48;
Sr = Xtest(j_min:j_max,:);

% Open loop
figure(2)
clf
plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');

for l = 1:10
    s = Sr(1,I.state_inx);
    S = s;
    for i = 1:size(Sr,1)
        a = Sr(i, I.action_inx);
        s = kdePropagate(s, a, r_max, Xtraining, I);
        S = [S; s];
    end   
    hold on
    plot(S(:,1),S(:,2),'.-');
    plot(S(1,1),S(1,2),'or','markerfacecolor','r');
    hold off
end
axis equal
legend('original path');
title('open loop');

%% Closed loop
figure(3)
clf
hold on
plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');

for i = 1:size(Sr,1)
    i
    s = Sr(i,I.state_inx);
    a = Sr(i, I.action_inx);
    S = zeros(1000, I.state_dim);
    for l = 1:1000
        l
        sp = kdePropagate(s, a, r_max, Xtraining, I);
%         plot([s(1) sp(1)],[s(2) sp(2)],'.-');
        S(i,:) = sp;
    end   
    sp = mean(S);
    plot([s(1) sp(1)],[s(2) sp(2)],'.-r','linewidth',2.5);
    drawnow;
end
plot(Sr(1,1),Sr(1,2),'or','markerfacecolor','m');
hold off
axis equal
legend('original path');
title('Closed loop (red=average)');


%%

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

