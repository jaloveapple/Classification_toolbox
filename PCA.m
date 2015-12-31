function [features, targets, UW, m] = PCA(features, targets, dimension, region)

%Reshape the data points using the principal component analysis
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	dimension		- Number of dimensions for the output data points
%	region			- Decision region vector: [-x x -y y number_of_points]
%
%Outputs
%	features		- New features
%	targets			- New targets
%	UW				- Reshape martix
%   m               - Original feature averages

[r,c] = size(features);

if (r < dimension),
   disp('Required dimension is larger than the data dimension.')
   disp(['Will use dimension ' num2str(r)])
   dimension = r;
end

%Calculate cov matrix and the PCA matrixes
m           = mean(features')';
S			= ((features - m*ones(1,c)) * (features - m*ones(1,c))');
[V, D]	    = eig(S);
W			= V(:,r-dimension+1:r)';
U			= S*W'*inv(W*S*W');

%Calculate new features
UW			= U*W;
features    = W*features;