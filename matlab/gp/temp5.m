clf;
imshow(IM);

for i = 1:10

v = ginput(2);

d(i) = norm(v(1,:)-v(2,:));

end

mean(d)