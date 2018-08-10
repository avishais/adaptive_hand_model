clear all
warning('off','all')

ps = parallel.Settings;
ps.Pool.AutoCreate = false;
% poolobj = gcp; % If no pool, do not create new one.

for mode = 1:8
    [Xtraining, Xtest, kdtree, I] = load_data(mode);
    
    Sr = Xtest;
    
    %%
    Np = 100;
    
    s = Sr(1, I.state_inx);
    P_prev = repmat(s, Np, 1);
    
    S = zeros(size(Sr,1)-1, I.state_dim);
    S(1,:) = mean(P_prev);
    P = cell(50,1);
    for i = 1:size(P,1)%size(S,1)-1
        disp(['Step: ', num2str(i)]);
        
        a = Sr(i, I.action_inx);
        
        P{i} = zeros(Np, I.state_dim);
        for k = 1:Np
            s = P_prev(k,:);
            [mu, sigma] = prediction(kdtree, Xtraining, s, a, I, 1);
            for j = 1:I.state_dim
                P{i}(k,j) = normrnd(mu(j),sigma(j));
            end
        end
        S(i+1,:) = mean(P{i});
        
        P_prev = P{i};
        %     drawnow;
        
    end
    
    %%
    figure(1)
    clf;
    plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
    
    hold on
    
    for i = 1:size(P,1)%size(S,1)-1
        
        ix = convhull(P{i}(:,1),P{i}(:,2));
        patch(P{i}(ix,1),P{i}(ix,2),'y')
        
        plot(P{i}(:,1),P{i}(:,2),'.k');
        plot(S(1:i,1),S(1:i,2),'-r','markersize',4,'markerfacecolor','r','linewidth',3);
        
    end
    
    plot(Sr(:,1),Sr(:,2),'-b','linewidth',3,'markerfacecolor','k');
    plot(S(1:i,1),S(1:i,2),'-r','markersize',4,'markerfacecolor','r','linewidth',3);
    
    title(['Feature conf. ' num2str(mode)]);
    
    hold off
    axis equal
    
    print(['var_' num2str(mode) '.png'],'-dpng','-r150');
    
    clear P Sr S
end


