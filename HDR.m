function [features, targets] = HDR(features, targets, New_dim, region)

%Reduce the dimensions of the data points using the hierarchical dimensionality reduction algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	New_dim 		- Number of dimensions for the output data points
%	region			- Decision region vector: [-x x -y y number_of_points]
%
%Outputs
%	features		- New features
%	targets			- New targets

[d,c] = size(features);
d_tag = New_dim;

if (d < New_dim),
   error('Required dimension is larger than the data dimension.')
end

for d_hat = d:-1:d_tag+1,
    %Compute R by equation 114 (DHS Chpter 10)
    sigma   = features * features';
    R       = zeros(size(sigma));
    for i = 1:d_hat,
        for j = 1:d_hat,
            if (i == j),
                R(j,j) = 0;
            else
                R(i,j) = sigma(i,j)/sqrt(sigma(i,i)*sigma(j,j));
            end
        end
    end
    R       = abs(R);
    
    %Find most correlated features
    [i,j] = find(max(max(R)) == R);
    i     = i(1);
    j     = j(1);
    
    %Merge dimentions i and j
    features(i,:) = (features(i,:) + features(j,:))/2;
    
    %Delete dimension j
    features      = features([1:j-1,j+1:end],:);
end

