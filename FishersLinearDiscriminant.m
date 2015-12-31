function [features, train_targets, w] = FishersLinearDiscriminant(train_features, train_targets, param, region)

%Reshape the data points using the Fisher's linear discriminant
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	param			- Unused
%	region			- Unused
%
%Outputs
%	features			- New features
%	targets			- New targets
%  w					- Weights vector

train_one  = find(train_targets == 1);
train_zero = find(train_targets == 0);

s0			  = cov(train_features(:,train_zero)',1);
m0			  = mean(train_features(:,train_zero)');
s1			  = cov(train_features(:,train_one)',1);
m1			  = mean(train_features(:,train_one)');

sw			  = s0 + s1;
w			  = inv(sw)*(m0-m1)';
features   = [w'*train_features; zeros(1,length(train_targets))]; %We add a dimension because the toolbox needs 2D data
