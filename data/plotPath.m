clear all

P = load('data_25_2.db');
% P = load('toyData.db');

xmax = max(P); xmax([1, 5]) = max(xmax([1, 5])); xmax([2, 6]) = max(xmax([2, 6]));
xmin = min(P); xmin([1, 5]) = min(xmin([1, 5])); xmin([2, 6]) = min(xmin([2, 6]));
P = (P-repmat(xmin, size(P,1), 1))./repmat(xmax-xmin, size(P,1), 1);

P = P(1:10,:);

% ix = 1;
% M = Q{ix}.data(:,:);
A = P(:,3:4);%M(:,Q{ix}.action_inx);
S = P(:,1:2);%M(:,Q{ix}.state_inx);
Sn = P(:,5:6);%M(:,Q{ix}.state_nxt_inx);

figure(1)
clf
plot(S(1,1),S(1,2),'sm','markersize',20,'markerfacecolor','m');
hold on
plot(S(:,1),S(:,2),'.-b','linewidth',3);
% plot(Sn(:,1),Sn(:,2),'x--r');

for i = 1:size(S,1)
    d = Action(A(i,:));
    quiver(S(i,1),S(i,2),d(1), d(2),0.01);
end

hold off

%%
% idx = rangesearch(P(:,1:2), [0.5 0.5], 0.01);
% S = P(idx{1},:);
% A = S(:,3:4);%M(:,Q{ix}.action_inx);
% Sp = S(:,1:2);%M(:,Q{ix}.state_inx);
% Sn = S(:,5:6);%M(:,Q{ix}.state_nxt_inx);
% 
% figure(1)
% clf
% hold on
% % plot(Sn(:,1),Sn(:,2),'x--r');
% 
% for i = 1:size(S,1)
%     plot([Sp(i,1) Sn(i,1)],[Sp(i,2) Sn(i,2)],'.-b','linewidth',3);
%     
%     d = Action(A(i,:));
%     quiver(Sp(i,1),Sp(i,2),d(1), d(2),0.1);
% end
% 
% hold off

%%
function d = Action(a)
% a = (a+0.06)/0.12;
if all(a==[0 0])
    d = [0 -1];
else if all(a==[1 1])
        d = [0 1];
    else if all(a==[1 0])
            d = [-1 0];
        else
            if all(a==[0 1])
                d = [1 0];
            end
        end
    end
end
end