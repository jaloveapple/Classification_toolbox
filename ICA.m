function [features, targets, W] = ICA(features, targets, params, region)

%Reshape the data points using the independent component analysis algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	params			- [Output dimension, Learning rate]
%	region			- Decision region vector: [-x x -y y number_of_points]
%
%Outputs
%	features			- New features
%	targets			- New targets
%	W					- Reshape martix

[r,c]			= size(features);
[dimension, eta] = process_params(params);

if (r < dimension),
   error('Output dimension cannot be larger than the input dimension')
end

%Whiten the data to zero mean and unit covariance
features = features - mean(features')'*ones(1,c);
[v, d]	= eig(cov(features',1));
Aw			= v*inv(sqrtm(d));
features = Aw'*features;

%Move data to the range of [-1,1]
features = (features - min(features')'*ones(1,c))./((max(features') - min(features'))'*ones(1,c));
features = features*2-1;

%Find the weight matrix
W			= randn(r);
for i = 1:c,
   y		= features(:,i);
   phi	= activation(y);
   dW		= eta*(eye(r) - phi*y')*W;
   
   %Break if algorithm diverges
   if (max(max(dW)) > 1e3),
       disp(['Algorithm diverged after ' num2str(i) ' iterations'])
       break
   end
   
   W		= W + dW;   
   
   %If the algorithm converged, exit
   if (max(max(W)) < 1e-2),
       disp(['Algorithm converged after ' num2str(i) ' iterations'])
       break
   end
end

%Take only the most influential outputs
power		= sum(abs(W)');
[m, in]	= sort(power);
W			= W(in(r-dimension+1:r),:);

%Calculate new features
features = W*features;


%End ICA

function phi = activation(y)
%Activation function for ICA
%phi=(3/4)*y.^11+(25/4)*y.^9+(-14/3)*y.^7+(-47/4)*y.^5+(29/4)*y.^3;
phi = 0.5*y.^5 + 2/3*y.^7 + 15/2*y.^9 + 2/15*y.^11 - 112/3*y.^13 + 128*y.^15 - 512/3*y.^17;