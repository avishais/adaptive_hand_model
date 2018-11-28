clear all
warning('off','all')

with_ml = 0;

T = cell(8,2);
t = 0;
test_num = 1;
data_source = '20';
w = [];%1.05 1.05 1 1 2 2 3 3]; % For cyl 25 and mode 8
N = 100;


for j = 1:2
    if j==1
        with_ml = 1;
    else
        with_ml = 0;
    end
    
    for mode = 1:8
        [Xtraining, Xtest, kdtree, I] = load_data(mode, w, test_num, data_source);
        Sr = Xtest;
        K = randi(size(Sr,1),N, 1);
        
        t = zeros(N,1);
        for i = 1:N
            disp([j mode i]);
            s = Sr(K(i), I.state_inx);
            a = Sr(K(i), I.action_inx);
            tic;
            if with_ml
                [s, s2] = prediction_mlgp(kdtree, Xtraining, s, a, I, 1);
            else
                [s, s2] = prediction(kdtree, Xtraining, s, a, I, 1);
            end
            t(i) = toc;
        end
        T{mode, j} = t;        
    end
end

disp(T);

save('time.mat','T');

%%

Tavg = zeros(8,2);
for i = 1:8
    for j = 1:2
        Tavg(i,j) = mean(T{i,j});
    end
end
disp([Tavg Tavg(:,2)./Tavg(:,1)]);