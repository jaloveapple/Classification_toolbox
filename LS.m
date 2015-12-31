function [D, w] = LS(train_features, train_targets, weights, region)

% Classify using the least-squares algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	Weights	- Weighted for weighted least squares (Optional)
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%	w			- Decision surface parameters

[Dim, Nf]       = size(train_features);
Dim             = Dim + 1;
train_features(Dim,:) = ones(1,Nf);

%Weighted LS or not?
switch length(weights),
case Nf + 1,
    %Ada boost form
    weights = weights(1:Nf);
case Nf,
    %Do nothing
otherwise
    weights = ones(1, Nf);
end

train_one  = find(train_targets == 1);
train_zero = find(train_targets == 0);

%Preprocess the targets
mod_train_targets = 2*train_targets - 1; 

w = inv((train_features .* (ones(Dim,1)*weights)) * train_features') * (train_features .* (ones(Dim,1)*weights)) * mod_train_targets';
%w = pinv(train_features * train_features') * train_features * mod_train_targets';

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D     = (w(1).*x + w(2).*y + w(3) > 0);
w		= w';
