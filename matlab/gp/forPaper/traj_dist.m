clear all
warning('off','all')

addpath('../');
addpath('../../../data/');

% files = dir(fullfile('../../../data/misc', ['*_seq_*.txt']));
files = dir(fullfile('../../../data/misc', ['ca_replay_20_test1_*.txt']));

files = struct2cell(files)';
files = sortrows(files, 1);
files = files(:,1);

files = [files; 'ca_20_test1.txt'];

n = size(files,1);

%%

mode = 8;
w = [5 5 1 1 2 2 3 3];
[Xtraining, Xtest, kdtree, I] = load_data(mode, w, 1, '20');

st_inx = 700;

%%

figure(1)
clf
hold on

E1 = [];
E2 = [];
data = cell(n,1);

it = 1;
for i = 1:n
    f = files{i,1};
    
    D = dlmread(['../../../data/misc/' f], ' ');
    
    % Clean data
    data{i} = process_data(D);
    
    data{i}.obj_pos = data{i}.obj_pos(st_inx:end,:);
    data{i}.act_load = data{i}.act_load(st_inx:end,:);
    data{i}.act_pos = data{i}.act_pos(st_inx:end,:);
    data{i}.T = data{i}.T(st_inx:end); data{i}.T = data{i}.T - data{i}.T(1);
    data{i}.n = length(st_inx:data{i}.n);
    
    %         j = 1;
    %         while data{i}.ref_vel
    
    plot(data{i}.obj_pos(:,1), data{i}.obj_pos(:,2), '-k');
    plot(data{i}.obj_pos(1,1), data{i}.obj_pos(1,2), 'ok','markerfacecolor','r');
    plot(data{i}.obj_pos(end,1), data{i}.obj_pos(end,2), 'ok','markerfacecolor','y');
    
    switch mode
        case 5
            E1 = [E1; data{i}.obj_pos(1,1:2), data{i}.act_load(1,:)];
        case 8
            E1 = [E1; data{i}.obj_pos(1,1:2), data{i}.act_pos(1,:), data{i}.act_load(1,:)];
    end
            
    E2 = [E2; data{i}.obj_pos(end,1:2)];
    
    if data{i}.n < data{it}.n
        it = i;
    end
end

e1 = mean(E1);
e2 = mean(E2);

plot(e1(1), e1(2), 'pk','markerfacecolor','r','markersize',14);
plot(e2(1), e2(2), 'pk','markerfacecolor','y','markersize',14);

hold off
axis equal

start_mean = mean(E1);
start_std = std(E1);

Rx = []; Ry = [];
for i = 1:n
    Rx = [Rx data{i}.obj_pos(1:data{it}.n,1)];
    Ry = [Ry data{i}.obj_pos(1:data{it}.n,2)];
end
Rm = [mean(Rx')' mean(Ry')'];
Rs = [std(Rx')' std(Ry')'];
Rx = [Rm(:,1)-Rs(:,1) Rm(:,1)+Rs(:,1)];
Ry = [Rm(:,2)-Rs(:,2) Rm(:,2)+Rs(:,2)];

%%

k = data{1}.n;
A = data{1}.ref_vel(st_inx:end,:);
A = (A-repmat(I.xmin(I.action_inx),k,1))./repmat(I.xmax(I.action_inx)-I.xmin(I.action_inx),k,1);

hold on
S = cell(10,1);
for i = 1%:10
    %
    
    % Sample from distribution
    s = zeros(1, length(start_mean));
    for ia = 1:length(start_mean)
        s(ia) = normrnd(start_mean(ia),start_std(ia));
    end
    s = (s-I.xmin(I.state_inx))./(I.xmax(I.state_inx)-I.xmin(I.state_inx));
    
    S{i} = zeros(k, I.state_dim);
    S{i}(1,:) = s;
    loss = 0;
    for j = 1:k-1
        a = A(j,:);
        disp(['Step: ' num2str([i j]) ', action: ' num2str(a)]);
        
%         [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);
        s = gdPropagate(s, a, kdtree, Xtraining, I);
        
        S{i}(j+1,:) = s;  
        
        if ~mod(j, 10)
            F = S{i}(1:j,:).*repmat(I.xmax(I.state_inx)-I.xmin(I.state_inx),j,1) + repmat(I.xmin(I.state_inx),j,1);
            plot(F(:,1),F(:,2),'.-m');
            drawnow;
        end
    end
    
    S{i} = S{i}.*repmat(I.xmax(I.state_inx)-I.xmin(I.state_inx),k,1) + repmat(I.xmin(I.state_inx),k,1);
    
    %
    plot(S{i}(:,1), S{i}(:,2), '--k');
    
    
end

%%
figure(1)
clf
hold on
for i = 1:n
    plot(data{i}.obj_pos(:,1), data{i}.obj_pos(:,2), '-k','linewidth',3);
    plot(data{i}.obj_pos(1,1), data{i}.obj_pos(1,2), 'ok','markerfacecolor','r');
    plot(data{i}.obj_pos(end,1), data{i}.obj_pos(end,2), 'ok','markerfacecolor','y');
end
plot(e1(1), e1(2), 'pk','markerfacecolor','r','markersize',14);
plot(e2(1), e2(2), 'pk','markerfacecolor','y','markersize',14);
for i = [1 4:6 8 9:10]
    plot(S{i}(:,1), S{i}(:,2), '--m','linewidth',3);
    plot(S{i}(end,1), S{i}(end,2), 'sk','markerfacecolor','c');
end
hold off
axis equal


Sx = []; Sy = [];
for i = [1 2 4:6 7 8 9:10]
    Sx = [Sx S{i}(:,1)];
    Sy = [Sy S{i}(:,2)];
end

Sm = [mean(Sx')' mean(Sy')'];
Ss = [std(Sx')' std(Sy')'];
Sx = [Sm(:,1)-Ss(:,1) Sm(:,1)+Ss(:,1)];
Sy = [Sm(:,2)-Ss(:,2) Sm(:,2)+Ss(:,2)];
% Sx = [min(Sx')' max(Sx')'];
% Sy = [min(Sy')' max(Sy')'];
    
%%    

figure(2)
clf
hold on
% fill([data{1}.T; flipud(data{1}.T)], [Sx(:,1); flipud(Sx(:,2))],'y');
% fill([data{1}.T; flipud(data{1}.T)], [Sy(:,1); flipud(Sy(:,2))],'y');
% plot(data{1}.T,Sm(:,1),':r',data{1}.T,Sm(:,2),':k','linewidth',2.5);

fill([data{it}.T; flipud(data{it}.T)], [Rx(:,1); flipud(Rx(:,2))],'b');
fill([data{it}.T; flipud(data{it}.T)], [Ry(:,1); flipud(Ry(:,2))],'b');
plot(data{it}.T,Rm(:,1),':r',data{it}.T,Rm(:,2),'-k','linewidth',2.5);


% errorbar(data{1}.T,Sm(:,1),Ss(:,1),':r');
% for i = 1:n
%     plot(data{i}.T, data{i}.obj_pos(:,1), '-r', data{i}.T, data{i}.obj_pos(:,2), '-b');
% end
% for i = [1 4:6 8 9:10]
%     plot(data{1}.T, S{i}(:,1), '--m', data{1}.T, S{i}(:,2), '--m');
% end
hold off
xlabel('Time (sec)');
ylabel('Position');
legend('pred. std','','pred. mean - x','pred. mean - y','test set - x','test set - y');
xlim([0 max(data{1}.T)-2]);

% print(['distributions25.png'],'-dpng','-r150');

%%

save('traj_dist_01');
