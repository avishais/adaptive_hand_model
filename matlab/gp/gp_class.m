classdef gp_class
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        mode
        w
        W
        Xtraining
        kdtree
        I
    end
    
    methods
        % Constructor
        function obj = gp_class(m)
            obj.mode = m;
            
            switch obj.mode
                case 1
                    obj.w = [3 3 1 1];
                case 2
                    obj.w = [3 3 1 1 1 1 1 1];
                case 3
                    obj.w = [3 3 1 1 1 1 1 1 1 1 3 3];
                case 4
                    obj.w = [];
                case 5
                    obj.w = [60 60 1 1 3 3];
                case 7
                    obj.w = [10 10 ones(1,14)];
                case 8
                    obj.w = [5 5 3 3 1 1 3 3];
            end
            
            obj = obj.load_data();
        end
        
        function obj = load_data(obj)
            
            data_source = '20';
            
            
            file = ['../../data/Ca_' data_source '_' num2str(obj.mode)];
            
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
                obj.W = diag(obj.w);
                obj.kdtree = createns(obj.Xtraining(:,[obj.I.state_inx obj.I.action_inx]), 'Distance',@obj.distfun);
            end
            
        end
        
        function D2 = distfun(obj, ZI,ZJ)
            
            if isempty(obj.W)
                obj.W = diag(ones(1,size(ZI,2)));
            end
            
            n = size(ZJ,1);
            D2 = zeros(n,1);
            for i = 1:n
                Z = ZI-ZJ(i,:);
                D2(i) = Z*obj.W*Z';
            end
            
        end
        
        function gprMdl = getPredictor(obj, s, a)
            
            [idx, ~] = knnsearch(obj.kdtree, [s a], 'K', 100);
            
            data_nn = obj.Xtraining(idx,:);
            
            gprMdl = cell(length(obj.I.state_nxt_inx),1);
            for i = 1:length(obj.I.state_nxt_inx)
                gprMdl{i} = fitrgp(data_nn(:,[obj.I.state_inx obj.I.action_inx]), data_nn(:,obj.I.state_nxt_inx(i)),'Basis','linear','FitMethod','exact','PredictMethod','exact');
            end
            
        end
        
        function [sp, sigma] = predict(obj, s, a)        
            
            gprMdl = obj.getPredictor(s, a);
            
            sp = zeros(1, length(obj.I.state_nxt_inx));
            sigma = zeros(1, length(obj.I.state_nxt_inx));
            for i = 1:length(obj.I.state_nxt_inx)
                [sp(i), sigma(i)] = predict(gprMdl{i}, [s a]);
            end
            
            
        end
        
        function x = normz(obj, x)
            x = (x-obj.I.xmin(1:length(x))) ./ (obj.I.xmax(1:length(x))-obj.I.xmin(1:length(x)));
        end
        
        function x = denormz(obj, x)
            x = x .* (obj.I.xmax(1:length(x))-obj.I.xmin(1:length(x))) + obj.I.xmin(1:length(x));
        end
        
    end
end

