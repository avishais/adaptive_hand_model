clear all
for mode = 1:8
    
    simple_gp;
    
    save(['gp_' num2str(mode) '.mat']);
    
end

%%
% mode = 8;
% load(['gp_' num2str(mode) '.mat']);
% 
% % Open loop
% figure(1)
% clf
% plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
% hold on
% plot(S(:,1),S(:,2),'.-r');
% plot(S(1,1),S(1,2),'or','markerfacecolor','r');
% hold off
% axis equal
% legend('original path','prediction');
% title('open loop');
% print(['im_' num2str(mode) '.png'],'-dpng','-r150');
% 
% figure(2)
% clf
% hold on
% plot(Sr(:,1),Sr(:,2),'o-b','linewidth',3,'markerfacecolor','k');
% for i = 1:size(Sr,1)
%         plot([Sr(i,1) Sp(i,1)],[Sr(i,2) Sp(i,2)],'.-');
% end
% plot(Sr(1,1),Sr(1,2),'or','markerfacecolor','m');
% hold off
% axis equal
% legend('original path');
% title('Closed loop');
% Mse = sqrt(sum);
% disp(['Error: ' num2str(Mse)]);