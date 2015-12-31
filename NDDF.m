function [D, g0, g1] = NDDF(train_features, train_targets, cost, region, test_feature)

% Classify using the normal density discriminant function
% Inputs:
% 	features			- Train features
%	targets			- Train targets
%	cost				- Cost for class 0 (Optional, Unused yet)
%	region			- Decision region vector: [-x x -y y number_of_points]
%	test_feature	- A test example (optional)
%
% Outputs
%	D			- Decision sufrace
%	g0, g1	- The discriminant function for the test example

[d, L] = size(train_features);
N		 = region(5);

%Estimate mean and covariance for each class
mu		= zeros(d,length(unique(train_targets)));
sigma	= zeros(d,d,length(unique(train_targets)));
p		= zeros(length(unique(train_targets)));

classes = unique(train_targets);
for i = 1:length(classes),
   indices			= find(train_targets == classes(i));
   mu(:,i)			= mean(train_features(:,indices)')';
   sigma(:,:,i)	= cov(train_features(:,indices)',1)';
   p(i)				= length(indices)/length(train_targets);
end

%Build a decision region for 2D, 2 class data
D 		= zeros(N);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

for i = 1:region(5),
   X		= [x(i,:) ; y(i,:)];
   g0		= -0.5*(X - mu(:,1)*ones(1,N))'*inv(squeeze(sigma(:,:,1)))*(X - mu(:,1)*ones(1,N)) - ...
   		   d/2*log(2*pi)-0.5*log(det(squeeze(sigma(:,:,1))))+log(p(1));
   g1		= -0.5*(X - mu(:,2)*ones(1,N))'*inv(squeeze(sigma(:,:,2)))*(X - mu(:,2)*ones(1,N)) - ...
   		   d/2*log(2*pi)-0.5*log(det(squeeze(sigma(:,:,2))))+log(p(2));
   D(i,:)= (diag(g0) < diag(g1))';
end
      
%If there is a test example, calculate g0 and g1 for it
if exist('test_feature'),
   X		= test_feature;
   g0		= -0.5*(X - mu(:,1)*ones(1,N))'*inv(squeeze(sigma(:,:,1)))*(X - mu(:,1)*ones(1,N)) - ...
   		   d/2*log(2*pi)-0.5*log(det(squeeze(sigma(:,:,1))))+log(p(1));
   g1		= -0.5*(X - mu(:,2)*ones(1,N))'*inv(squeeze(sigma(:,:,2)))*(X - mu(:,2)*ones(1,N)) - ...
   		   d/2*log(2*pi)-0.5*log(det(squeeze(sigma(:,:,2))))+log(p(2));
end   