clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

test_num = 3;
mode = 11;
w = 1;
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, 'all');


G25 = Xtraining;%(Xtraining(:,7)==0.625,:);
G35 = Xtraining;%(Xtraining(:,7)==0.8750,:);

c25 = 0; 
for i = 1:size(G25,1)
    c25 = c25 + all(G25(i,6:7)==1);
end
c35 = 0;
for i = 1:size(G35,1)
    c35 = c35 + all(G35(i,6:7)==1);
end


disp([c25 c35]);

