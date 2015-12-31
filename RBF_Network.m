function [D, mu, Wo] = RBF_Network(train_features, train_targets, Nh, region)

% Classify using a backpropagation network with a batch learning algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	Nh      - Number of hidden units
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   mu          - Hidden unit locations
%   Wo          - Output unit weights

[Ni, M] = size(train_features);
sigma   = sqrt(Ni/sqrt(2*M));            %Variance of the gaussians

%First, find locations for the hidden unit centers using k-means
[mu, center_targets, label] = k_means(train_features, train_targets, Nh, region, 0);

%Compute the activation for each feature at each center
Phi = zeros(Nh, M);
for i = 1:Nh,
    Phi(i,:) = 1/(2*pi*sigma^2)^(Ni/2)*exp(-sum((train_features-mu(:,i)*ones(1,M)).^2)/(2*sigma^2));
end

%Now, find the hidden to output weights
Wo  = (pinv(Phi)'*(train_targets*2-1)')';

%Build a decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:)]';
Phi         = zeros(Nh, N^2);
for i = 1:Nh,
    Phi(i,:) = 1/(2*pi*sigma^2)^(Ni/2)*exp(-sum((flatxy-mu(:,i)*ones(1,N^2)).^2)/(2*sigma^2));
end
D           = reshape(Wo*Phi>0,N,N);