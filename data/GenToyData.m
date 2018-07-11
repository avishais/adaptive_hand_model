clear all

if 0
    data = record(1e5);
    dlmwrite('toyData.db', data, ' ');
else
    So = [-40 -40];
    Ai = [ones(10, 1)*4; ones(10, 1)*1; ones(20,1)*2; ones(10,1)*3];
    data = record(length(Ai), So, Ai);
    dlmwrite('toyDataPath.db', data, ' ');
end
    

function data = record(N, So, Ar)

aflag = 1;
if nargin==1
    So = randPos();
    aflag = 0;
end

x = So;
data = zeros(N,6);
i = 1;
vel = 400;
A = [-1 1; -1 -1; 1 -1; 1 1];
while i <= N
    simu(x);
    
    if mod(i, 100)==0
        x = randPos();
    end
    
    % Choose action
    if aflag
        a = A(Ar(i),:)*vel;
    else
        a = A(randi(4),:)*vel;
    end
    
    x_next = prop(x, a);
    
    if any(abs(x_next) > 100 )
        x = randPos();
        continue;
    end
    
    data(i,:) = [x a x_next];
    x = x_next;
    
    i = i + 1;
end

end


function simu(x)

figure(1)
clf
plot(x(1),x(2),'ok','markersize',15,'markerfacecolor','r');
axis equal
axis([-100 100 -100 100]);
drawnow;
end

function x = randPos()
x = rand(1,2) * 200 - 100;
end


function x_next = prop(x, u)

dt = 0.01;

% f = u(1)*[cos(u(2)); sin(u(2))];
if u(1) == u(2) && u(1) < 0
    f = [0 -abs(u(1))];
else if u(1) == u(2) && u(1) > 0
        f = [0 abs(u(1))];
    else if u(1) ~= u(2) && u(1) > 0
            f = [-abs(u(1)) 0];
        else if u(1) ~= u(2) && u(1) < 0
                f = [abs(u(1)) 0];
            end
        end
    end
end
        

x_next = x + f*dt + normrnd(0, 0.8, [1, 2]); %0.8

end

