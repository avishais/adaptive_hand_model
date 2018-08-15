[Xtraining, Xtest, kdtree, I] = load_data(5, 1, 1);
X = Xtraining(:,[I.state_inx I.action_inx]);

r = 0.05;

%%
for i = 1:100
    
    [x,y] = ginput(1);
    idx = knnsearch(X(:,3:4), [x y]);
    x = X(idx,:);  
    
    figure(2)
    subplot(211)
    plot(X(:,1),X(:,2),'.')
    hold on
    plot(x(1),x(2),'om','markerfacecolor','m');
    hold off
    circle(x(1),x(2),r);
    axis equal
    axis([0 1 0 1]);
    
    subplot(212)
    plot(X(:,3),X(:,4),'.')
    hold on
    plot(x(3),x(4),'om','markerfacecolor','m');
    hold off
    circle(x(3),x(4),r);
    axis equal
    axis([0 1 0 1]);
    
    id = rangesearch(X, x, r); id = id{1};
    disp(length(id));
        
end

%% Find average neighbors in data

N = 1e4;
L = zeros(N,1);
max_nn = 0;
min_nn = 1e5;
for i = 1:N
    disp(['Step: ' num2str(i)]);
    x = X(randi(size(X,1)),:);
    
    id = rangesearch(X, x, r); id = id{1};
    L(i) = length(id)-1;
end
disp(['Average number of nn: ' num2str(mean(L))]);
disp(['Max. number of nn: ' num2str(max((L)))]);
disp(['Min. number of nn: ' num2str(min(L))]);

figure(1)
clf
hist(L,100);



%%

function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit, 'y');
hold off
end