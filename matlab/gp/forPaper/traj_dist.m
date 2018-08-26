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

% hold on
S = cell(100,1);
for i = 1:100
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
        
%         if ~mod(j, 10)
%             F = S{i}(1:j,:).*repmat(I.xmax(I.state_inx)-I.xmin(I.state_inx),j,1) + repmat(I.xmin(I.state_inx),j,1);
%             plot(F(:,1),F(:,2),'.-m');
%             drawnow;
%         end
    end
    
    S{i} = S{i}.*repmat(I.xmax(I.state_inx)-I.xmin(I.state_inx),k,1) + repmat(I.xmin(I.state_inx),k,1);
    
    %
%     plot(S{i}(:,1), S{i}(:,2), '--k');

    save('traj_dist_20');
    
    
end

%%

K = [1:12 14:23 25:31 33:35 37:41 43:48 51:58 61:63 65:66 68:69 71:75];

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
for i = K
    plot(S{i}(:,1), S{i}(:,2), '--m','linewidth',3);
    plot(S{i}(end,1), S{i}(end,2), 'sk','markerfacecolor','c');
end
hold off
axis equal


Sx = []; Sy = [];
for i = K
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

px2mm = 0.2621;

U = repmat(I.base_pos*px2mm, data{1}.n, 1);
Ut = repmat(I.base_pos*px2mm, data{it}.n, 1);


figure(2)
clf
subplot(211)
hold on
fill([data{1}.T; flipud(data{1}.T)], [Sx(:,1)*px2mm+U(:,1); flipud(Sx(:,2)*px2mm+U(:,1))],'y');
fill([data{it}.T; flipud(data{it}.T)], [Rx(:,1)*px2mm+Ut(:,1); flipud(Rx(:,2)*px2mm+Ut(:,1))],'-b');
plot(data{1}.T,Sm(:,1)*px2mm+U(:,1),':k','linewidth',2.5);
plot(data{it}.T,Rm(:,1)*px2mm+Ut(:,1),'-k','linewidth',2.5);
% plot([data{1}.T; flipud(data{1}.T)], [Sx(:,1); flipud(Sx(:,2))],'-k','linewidth',4);
hold off
set(gca, 'FontSize', 12)
xlabel('Time (sec)','fontsize', 17);
ylabel('Position - x axis (mm)','fontsize', 17);
% legend({'pred. std.','ground truth std.','pred. mean','ground truth mean'},'location','northwest','fontsize',14);
xlim([0 max(data{it}.T)-2]);

subplot(212)
hold on
fill([data{1}.T; flipud(data{1}.T)], [Sy(:,1)*px2mm+U(:,2); flipud(Sy(:,2)*px2mm+U(:,2))],'y');
fill([data{it}.T; flipud(data{it}.T)], [Ry(:,1)*px2mm+Ut(:,2); flipud(Ry(:,2)*px2mm+Ut(:,2))],'b');
plot(data{1}.T,Sm(:,2)*px2mm+U(:,2),':k','linewidth',2.5);
plot(data{it}.T,Rm(:,2)*px2mm+Ut(:,2),'-k','linewidth',2.5);
% plot([data{1}.T; flipud(data{1}.T)], [Sy(:,1); flipud(Sy(:,2))],'-k','linewidth',3);
hold off
set(gca, 'FontSize', 12)
xlabel('Time (sec)','fontsize', 17);
ylabel('Position - y axis (mm)','fontsize', 17);
legend({'pred. std.','ground truth std.','pred. mean','ground truth mean'},'location','northwest','fontsize',14);
xlim([0 max(data{it}.T)-2]);

% errorbar(data{1}.T,Sm(:,1),Ss(:,1),':r');
% for i = 1:n
%     plot(data{i}.T, data{i}.obj_pos(:,1), '-r', data{i}.T, data{i}.obj_pos(:,2), '-b');
% end
% for i = [1 4:6 8 9:10]
%     plot(data{1}.T, S{i}(:,1), '--m', data{1}.T, S{i}(:,2), '--m');
% end


print(['distributions20.png'],'-dpng','-r150');

%%

save('traj_dist_20');
