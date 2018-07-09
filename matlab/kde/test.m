clear all

num_net = 1;
Q = load(['../../data/data' num2str(num_net) '.mat'], 'Q');
Q = Q.Q;

Xtraining = [];
for i = 1:size(Q,1)-1
    Xtraining = [Xtraining; Q{i}.data(:,1:6)];    
end
xmax = max(Xtraining);
xmin = min(Xtraining);
Xtraining = (Xtraining-repmat(xmin, size(Xtraining,1), 1))./repmat(xmax-xmin, size(Xtraining,1), 1);

Xtest = Q{end}.data(2:end,1:6);
Xtest = (Xtest-repmat(xmin, size(Xtest,1), 1))./repmat(xmax-xmin, size(Xtest,1), 1);

%% Max motion

r_max = 0;
for i = 1:size(Xtraining,1)
    r = norm(Xtraining(1:2)-Xtraining(5:6));
    if r > r_max
        r_max = r;
    end        
end

%%

j = 4;

figure(1)
clf
Sr = Xtest(1:j,1:2);
plot(Sr(:,1),Sr(:,2),'.-b');

for l = 1:10
    s = Xtest(1,1:2);
    S = s;
    for i = 1:j%size(Xtest,1)
        a = Xtest(i, 3:4);
        s = kdePropagate(s, a, r_max, Xtraining);
        S = [S; s];
    end
    
    hold on
    plot(S(:,1),S(:,2),'.-k');
    plot(S(1,1),S(1,2),'or','markerfacecolor','r');
    hold off
    drawnow;
end


