f = 'ca_25_test2.txt';
D = dlmread(['./ca/' f], ' ');

S = [D(:,18:19) D(:,4:5)];
S(:,1:2) = MovingAvgFilter(S(:,1:2));

A = D(:,6:7);

plot(S(:,1),S(:,2));



function y = MovingAvgFilter(x, windowSize)
if nargin==1
    windowSize = 13;
end
y = x;
w = floor(windowSize/2);
for j = 1:size(x,2)
    for i = w+1:size(x,1)-w-1
        y(i,j) = sum(x(i-w:i+w,j))/length(i-w:i+w);
    end
end
end