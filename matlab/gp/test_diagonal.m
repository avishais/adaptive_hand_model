clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 1;
mode = 8;
% w = [1.05 1.05 1 1 2 2 3 3]; % For cyl 25 and mode 8
w = [5 5 1 1 2 2 3 3];
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, '20');

Sr = Xtest;

id = 1;
s = Sr(1,I.state_inx);

A = [1 1; 1 0; 0 1; 0 0];
Ad = [ 1 0.5; 0 0.5; 0.5 1; 0.5 0];

S = zeros(size(A,1),I.state_dim);
Sd = zeros(size(Ad,1),I.state_dim);
for i = 1:size(A,1)
    S(i,:) = prediction(kdtree, Xtraining, s, A(i,:), I, 1);
    Sd(i,:) = prediction(kdtree, Xtraining, s, Ad(i,:), I, 1);
end

figure(1)
clf
plot(s(1),s(2),'ok','markerfacecolor','y');
hold on
plot(S(:,1),S(:,2),'ok','markerfacecolor','g');
plot(Sd(:,1),Sd(:,2),'pk','markerfacecolor','r');
hold off