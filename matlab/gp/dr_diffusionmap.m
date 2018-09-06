function v = dr_diffusionmap(TS, dim)

N = size(TS,1);
data = TS;

%% Changing these values will lead to different nonlinear embeddings
knn    = ceil(0.01*N); % each patch will only look at its knn nearest neighbors in R^d
sigma2 = 1000; % determines strength of connection in graph... see below

%% now let's get pairwise distance info and create graph 
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

% eigVecIdx = nchoosek(2:4,2);
% for i = 1:size(eigVecIdx,1)
%     figure,scatter(v(:,eigVecIdx(i,1)),v(:,eigVecIdx(i,2)),20,cmap)
%     title('Nonlinear embedding');
%     xlabel(['\phi_',num2str(eigVecIdx(i,1))]);
%     ylabel(['\phi_',num2str(eigVecIdx(i,2))]);
% end

%%
% figure(4)
% clf
% scatter(v(:,2),v(:,3),20,cmap)
% title('Nonlinear embedding')