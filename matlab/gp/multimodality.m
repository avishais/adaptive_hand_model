clear all
warning('off','all')

UseToyData = false;

for mode = 1%:8
    [Xtraining, Xtest, kdtree, I] = load_data(mode);
    
    %%
    
    % A = [0 0; 1 1; 1 0; 0 1];
    
%     k = 68731;
%     k = 70961;
    k = randi(size(Xtraining,1));
    
    xa = Xtraining(k, [I.state_inx I.action_inx]);
    [idx, d] = knnsearch(kdtree, xa, 'K', 100+1);
    
    dnn = Xtraining(idx(2:end),:);
    
    G = zeros(size(dnn,1),1);
    N = zeros(size(dnn,1),1);
    for j = 1:size(dnn,1)
        v = dnn(j, I.state_nxt_inx) - dnn(j, I.state_inx);
        G(j) = rad2deg(atan2(v(2),v(1)));
        N(j) = norm(v);
    end
    
    G(G<0) = G(G<0) + 360;
    
    %%
    figure(1)
    clf
    subplot(2,2,1:2)
    hist(G,100);
    xlim([0 360]);
    title(['Feature conf. ' num2str(mode)]);
    
    % subplot(2,2,2)
    % hist(N,100);
    % xlim([0 0.1]);
    % title(['Feature conf. ' num2str(mode)]);
    
    
    subplot(223)
    hold on
    plot(dnn(:,1),dnn(:,2),'ob','markerfacecolor','y');
    plot(xa(1),xa(2),'pk','markerfacecolor','c','markersize',14);
    for i = 1:size(dnn,1)
        d = what_action(dnn(i,I.action_inx));
        quiver(dnn(i,1),dnn(i,2), d(1), d(2),0.002,'k');
    end
    hold off
    title('Obj. space');
    
    if mode==5 || mode==7 || mode==8
        if mode==5
            ix = 3:4;
        end
        if mode==7
            ix = 13:14;
        end
        if mode==8
            ix = 5:6;
        end
        subplot(224)
        hold on
        plot(dnn(:,ix(1)),dnn(:,ix(2)),'ok','markerfacecolor','y');
        plot(xa(ix(1)),xa(ix(2)),'pk','markerfacecolor','c','markersize',14);
        hold off
        title('Load space');
    end
    
    print(['mm' num2str(mode) '.png'],'-dpng','-r150');
    
end
%%
function d = what_action(a)
if all(a==[0 0])
    d = [0 -1];
else if all(a==[1 1])
        d = [0 1];
    else if all(a==[0 1])
            d = [1 0];
        else
            if all(a==[1 0])
                d = [-1 0];
            end
        end
    end
end
end