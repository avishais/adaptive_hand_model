clear all
clc


GP = gp_class(5);

s = [74.574 -417.29 44 -38];

% AA = [0, -0.4; -0.4, 0; -0.2, -0.2; -0.2, 0.2; 0.2, -0.2; 0.2, 0.2; 0, 0.4; 0.4, 0];
AA = [-0.2, 0.2; 0.2, -0.2; 0.2, 0.2; 0, 0.4; 0.4, 0];

MinS = [-171.0672 -430.5835  -163.0000 -556.0000];
MaxS = [293.8557 -176.4176  601.0000   16.0000];

figure(1)
clf
plot(s(1),s(2),'pb','markerfacecolor','b');
xlim([MinS(1) MaxS(1)]);
ylim([MinS(2) MaxS(2)]);
axis equal

S = s;
A = [];
S2 = [];
k = 1;
for i = 1:10
    a = AA(randi(size(AA,1)),:);
    n = randi([20, 400]);
    
    for j = 1:n
        
%         sa = GP.normz([s a]);       
        nn = -1;%GP.getNN(sa(1:4), sa(5:6), 12);
%         if nn < 5000
%             break;
%         end
        
        [s_next, sigma] = GP.predict(s, a);
        
%         s_next = GP.denormz(s_next);
        
        if any(s_next > MaxS) || any(s_next < MinS)
            break;
        end
        
        S = [S; s_next];
        S2 = [S2; sigma];
        A = [A; a];
        
        s = s_next;  

        if ~mod(k,10)
            hold on
            plot(S(:,1),S(:,2),'.-m');
            hold off
            drawnow;
        end
        k = k + 1;
        disp([num2str(k) ': Action - [' num2str(a(1)) ' ' num2str(a(2)) '], step ' num2str(j) ' out of ' num2str(n) ', number of NN - ' num2str(nn)]);
    end
       
end

dlmwrite('actionPath.txt', A, ' ');
dlmwrite('stateMeanPath.txt', S, ' ');
dlmwrite('stateStdPath.txt', S2, ' ');


