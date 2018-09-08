classdef gp_class < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        mode
        w
        We
        Xtraining
        kdtree
        I
        euclidean
%         predictServer
    end
    
    methods
        % Constructor
        function obj = gp_class(m)
%             rosinit
%             obj.predictServer = rossvcserver('/predictWithState', 'gp_predict/StateAction2State',@obj.predictStateCallback)
            
            obj.mode = m;
            obj.w = [];
            
            obj.euclidean = false;
            
            obj = obj.load_data();
            disp("Finished constructor")
        end
        
%         function predictStateCallback(obj)
%             exampleHelperROSCreateSampleNetwork
        
        function obj = load_data(obj)
            
            data_source = '20';
            
            
%             file = ['../../data/Ca_' data_source '_' num2str(obj.mode)];
            file = ['/home/akimmel/Documents/ICRA_2019_adaptive/Cc_' data_source '_' num2str(obj.mode)];
            
            
            if strcmp(data_source, '20')
                D = load([file '.mat'], 'Q', 'Xtraining');
                Q = D.Q;
                
                obj.Xtraining = D.Xtraining;
            else
                error('Wrong data source!');
            end
            
            igx = 1;
            obj.I.action_inx = Q{igx}.action_inx;
            obj.I.state_inx = Q{igx}.state_inx;
            obj.I.state_nxt_inx = Q{igx}.state_nxt_inx;
            obj.I.state_dim = length(obj.I.state_inx);
            
            xmax = max(obj.Xtraining);
            xmin = min(obj.Xtraining);
            
            for i = 1:obj.I.state_dim
                id = [i i+obj.I.state_dim+length(obj.I.action_inx)];
                xmax(id) = max(xmax(id));
                xmin(id) = min(xmin(id));
            end
            obj.Xtraining = (obj.Xtraining-repmat(xmin, size(obj.Xtraining,1), 1))./repmat(xmax-xmin, size(obj.Xtraining,1), 1);
            
            obj.I.xmin = xmin;
            obj.I.xmax = xmax;
            
            if isempty(obj.w)
                obj.kdtree = createns(obj.Xtraining(:,[obj.I.state_inx obj.I.action_inx]), 'NSMethod','kdtree','Distance','euclidean');
            else
                obj.We = diag(obj.w);
                obj.kdtree = createns(obj.Xtraining(:,[obj.I.state_inx obj.I.action_inx]), 'Distance',@obj.distfun);
            end
            
        end
        
        function D2 = distfun(obj, ZI,ZJ)
            
            if isempty(obj.We)
                obj.We = diag(ones(1,size(ZI,2)));
            end
            
            n = size(ZJ,1);
            D2 = zeros(n,1);
            for i = 1:n
                Z = ZI-ZJ(i,:);
                D2(i) = Z*obj.We*Z';
            end
            
        end
        
        function gprMdl = getPredictor(obj, s, a)
            
            if obj.euclidean
                [idx, ~] = knnsearch(obj.kdtree, [s a], 'K', 100);
                data_nn = obj.Xtraining(idx,:);
            else 
                data_nn =  obj.diffusion_metric([s a]);
            end
            
            gprMdl = cell(length(obj.I.state_nxt_inx),1);
            for i = 1:length(obj.I.state_nxt_inx)
                gprMdl{i} = fitrgp(data_nn(:,[obj.I.state_inx obj.I.action_inx]), data_nn(:,obj.I.state_nxt_inx(i)),'Basis','linear','FitMethod','exact','PredictMethod','exact');
            end
            
        end
        
        function [sp, sigma] = predict(obj, s, a)
            
            gprMdl = obj.getPredictor(s, a);
            
            sp = zeros(1, length(obj.I.state_nxt_inx));
            sigma = zeros(1, length(obj.I.state_nxt_inx));
            
            sa = obj.normz([s,a]);
            
            for i = 1:length(obj.I.state_nxt_inx)
                [sp(i), sigma(i)] = predict(gprMdl{i}, sa);
            end
            
            sigma_minus = obj.denormz(sp - sigma);
            
            sp = obj.denormz(sp);
            sigma = sp -sigma_minus;
            
        end
        
        function v = dr_diffusionmap(obj, TS, dim)
            
            N = size(TS,1);
            data = TS;
            
            % Changing these values will lead to different nonlinear embeddings
            knn    = ceil(0.03*N); % each patch will only look at its knn nearest neighbors in R^d
            sigma2 = 100; % determines strength of connection in graph... see below
            
            % now let's get pairwise distance info and create graph
            m                = size(data,1);
            dt               = squareform(pdist(data));
            [srtdDt,srtdIdx] = sort(dt,'ascend');
            dt               = srtdDt(1:knn+1,:);
            nidx             = srtdIdx(1:knn+1,:);
            
            % nz   = dt(:) > 0;
            % mind = min(dt(nz));
            % maxd = max(dt(nz));
            
            % compute weights
            tempW  = exp(-dt.^2/sigma2);
            
            % build weight matrix
            i = repmat(1:m,knn+1,1);
            W = sparse(i(:),double(nidx(:)),tempW(:),m,m);
            W = max(W,W'); % for undirected graph.
            
            % The original normalized graph Laplacian, non-corrected for density
            ld = diag(sum(W,2).^(-1/2));
            DO = ld*W*ld;
            DO = max(DO,DO');%(DO + DO')/2;
            
            % get eigenvectors
            [V,D] = eigs(DO,10,'la');
            
            v = V(:,1:dim);
        end
        
        function data_nn = diffusion_metric(obj, sa)
            
            [idx, ~] = knnsearch(obj.kdtree, sa, 'K', 1001);
            data = obj.Xtraining(idx,:);
            
            data_reduced = obj.dr_diffusionmap(data(:,[obj.I.state_inx obj.I.action_inx]), 3);
            sa_reduced_closest = data_reduced(1,:);
            data_reduced = data_reduced(2:end,:);
            
            idx_new = knnsearch(data_reduced, sa_reduced_closest, 'K', 100);
            
            data_nn = data(idx_new,:);
        end
        
        function num_neighbors = getNN(obj, s, a, r)
            sa = obj.normz([s,a]);
            id = rangesearch(obj.kdtree, sa,r);
            id = id{1};
            num_neighbors = length(id);
        end
        
        function x = normz(obj, x)
            x = (x-obj.I.xmin(1:length(x))) ./ (obj.I.xmax(1:length(x))-obj.I.xmin(1:length(x)));
        end
        
        function x = denormz(obj, x)
            x = x .* (obj.I.xmax(1:length(x))-obj.I.xmin(1:length(x))) + obj.I.xmin(1:length(x));
        end
        
    end
end

