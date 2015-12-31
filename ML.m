function D = ML(train_features, train_targets, AlgorithmParameters, region)

% Classify using the maximum-likelyhood algorithm
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Dummy		- Unused
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

train_one  = find(train_targets == 1);
train_zero = find(train_targets == 0);

%Estimate mean and covariance for class 0
param_struct.m0 = mean(train_features(:,train_zero)');
param_struct.s0 = cov(train_features(:,train_zero)',1);
param_struct.p0 = length(train_zero)/length(train_targets);

%Estimate mean and covariance for class 1
param_struct.m1 = mean(train_features(:,train_one)');
param_struct.s1 = cov(train_features(:,train_one)',1);

param_struct.w0 = 1;
param_struct.w1 = 1;

%Find decision region
D		= decision_region(param_struct, region);
