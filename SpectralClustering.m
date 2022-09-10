function [group, eigengap] = SpectralClustering(W, NUMC)
%SPECTRALCLUSTERING Executes spectral clustering algorithm


% calculate degree matrix
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = D - W;
k = max(NUMC);
% compute normalized Laplacian if needed

% avoid dividing by zero
degs(degs == 0) = eps;
% calculate D^(-1/2)
D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
% calculate normalized Laplacian
L = D * L * D;

% compute the eigenvectors corresponding to the k smallest
% eigenvalues
[U, eigenvalue] = eigs(L, k, eps);
[a,b] = sort(diag(eigenvalue),'ascend');
eigenvalue = eigenvalue(:,b);
U = U(:,b);
eigengap = abs(diff(diag(eigenvalue)));
U = U(:,1:k);
% in case of the Jordan-Weiss algorithm, we need to normalize
% the eigenvectors row-wise
%U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
%U = U./repmat(sqrt(sum(U.^2,2)),1,size(U,2));

flag =0;
for ck = NUMC
    Cindex = find(NUMC==ck);
    UU = U(:,1:ck);
    UU = UU./repmat(sqrt(sum(UU.^2,2)),1,size(UU,2));
    [EigenvectorsDiscrete]=discretisation(UU);
    [~,temp] = max(EigenvectorsDiscrete,[],2);
%     for i = 1 : ck
%         initcenter(i,:) = mean(UU(temp==i,:));
%     end
    
    Cluster{Cindex} = temp;
end


if length(NUMC)==1
    group=Cluster{1};
else
    group = Cluster;
end

function [EigenvectorsDiscrete,EigenVectors]=discretisation(EigenVectors)
% 
% EigenvectorsDiscrete=discretisation(EigenVectors)
% 
% Input: EigenVectors = continuous Ncut vector, size = ndata x nbEigenvectors 
% Output EigenvectorsDiscrete = discrete Ncut vector, size = ndata x nbEigenvectors
%
% Timothee Cour, Stella Yu, Jianbo Shi, 2004

[n,k]=size(EigenVectors);

vm = sqrt(sum(EigenVectors.*EigenVectors,2));
EigenVectors = EigenVectors./repmat(vm+eps,1,k);

R=zeros(k);
% R(:,1)=EigenVectors(1+round(rand(1)*(n-1)),:)';
 R(:,1)=EigenVectors(round(n/2),:)';
%R(:,1)=EigenVectors(n,:)';
c=zeros(n,1);
for j=2:k
    c=c+abs(EigenVectors*R(:,j-1));
    [minimum,i]=min(c);
    R(:,j)=EigenVectors(i,:)';
end

lastObjectiveValue=0;
exitLoop=0;
nbIterationsDiscretisation = 0;
nbIterationsDiscretisationMax = 20;%voir
while exitLoop== 0 
    nbIterationsDiscretisation = nbIterationsDiscretisation + 1 ;   
    EigenvectorsDiscrete = discretisationEigenVectorData(EigenVectors*R);
    [U,S,V] = svd(EigenvectorsDiscrete'*EigenVectors+eps,0);    
    NcutValue=2*(n-trace(S));
    
    if abs(NcutValue-lastObjectiveValue) < eps | nbIterationsDiscretisation > nbIterationsDiscretisationMax
        exitLoop=1;
    else
        lastObjectiveValue = NcutValue;
        R=V*U';
    end
end

function Y = discretisationEigenVectorData(EigenVector)
% Y = discretisationEigenVectorData(EigenVector)
%
% discretizes previously rotated eigenvectors in discretisation
% Timothee Cour, Stella Yu, Jianbo Shi, 2004

[n,k]=size(EigenVector);


[Maximum,J]=max(EigenVector');
 
Y=sparse(1:n,J',1,n,k);    
% Y = J';