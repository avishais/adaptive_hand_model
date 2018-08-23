mode = 1;
switch mode
    case 1
        q = [1 1 1 1];
        r = 0.01;
    case 5
        w = [30 30 1 1 10 10];
        r = 0.05;
    case 8
        w = [5 5 1 1 2 2 3 3];
        r = 0.1;
end
[Xtraining, Xtest, kdtree, I] = load_data(5, 1, 1,'20');
X = Xtraining(:,[I.state_inx I.action_inx]);

r = 0.05;

%% Find average neighbors in data

N = 1e2;
L = zeros(N,1);
for i = 1:N
    disp(['Step: ' num2str(i)]);
    k = randi(size(X,1));
    x = X(k,:);
    
    id = rangesearch(kdtree, x, r); id = id{1};
    L(i) = length(id)-1;
end
disp(['Average number of nn: ' num2str(mean(L))]);
disp(['Max. number of nn: ' num2str(max((L)))]);
disp(['Min. number of nn: ' num2str(min(L))]);

figure(1)
clf
hist(L, 100);
