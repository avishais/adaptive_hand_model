% clf;
% imshow(IM);

for i = 1:10

v = ginput(2);

d(i) = norm(v(1,:)-v(2,:));

end

mean(d)

%%
d = 0;

v = ginput(1);
for i = 2:30
    
x = ginput(1);

d = d + norm(x-v);
v = x;

end

%%
d1 = 0;
d2 = 0;
for i = 2:size(SI,1)
d1(i) = d1(i-1) + norm(SI(i,1:2)-SI(i-1,1:2));
d2(i) = d2(i-1) + norm(SRI(i,1:2)-SRI(i-1,1:2));
end

H = [SRI(:,1:2) SI(:,1:2) d1' d2'];

figure(1)
clf
plot(d2,SRI(:,1),'r');
hold on
plot(d1,SI(:,1),'b');
hold off

