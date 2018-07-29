clear all

Xr = load('c_25_7_rerun_processed.txt');
X = load('c_25_7_processed.txt');

% xmax = [400.261137478864,-136.433600218537,494,20,0.0600000000000000,0.0600000000000000,400.261137478864,-136.433600218537,494,20];
% xmin = [-229.321049989905,-426.300600902615,-13,-516,-0.0600000000000000,-0.0600000000000000,-229.321049989905,-426.300600902615,-13,-516];


%%

figure(1)
clf
hold on
plot(X(:,1),X(:,2),'r');
plot(Xr(:,1),Xr(:,2),'b');
% for i = 1:size(X,1)
%     d = what_action(X(i,4:5));
%     quiver(X(i,1),X(i,2), d(1), d(2),1,'k');
% end
% for i = 1:size(Xr,1)
%     d = what_action(Xr(i,4:5));
%     quiver(Xr(i,1),Xr(i,2), d(1), d(2),1,'k');
% end
hold off
% axis equal
legend('original','rerun');

figure(2)
clf
subplot(211)
hold on
plot(X(:,1),'r');
plot(Xr(:,1),'b');
hold off
subplot(212)
hold on
plot(X(:,2),'r');
plot(Xr(:,2),'b');
hold off
% axis equal
legend('original','rerun');


%%

function d = what_action(a)
if all(a==[0 0])
    d = [0 0];
    return;
end
if all(a==[0.06 0.06])
    d = [0 1];
    return;
end
if all(a==[-0.06 -0.06])
    d = [0 -1];
    return;
end
if all(a==[-0.06 0.06])
    d = [-1 0];
    return;
end
if all(a==[0.06 -0.06])
    d = [1 0];
    return;
end

end