function D = ML_diag(train_features, train_targets, AlgorithmParameters, region)

% Classify using the maximum likelyhood algorithm with diagonal covariance matrices
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
m0 = mean(train_features(:,train_zero)');
s0 = zeros(2);
s0(1,1) = var(train_features(1,train_zero) - m0(1));
s0(2,2) = var(train_features(2,train_zero) - m0(2));
P0 = length(train_zero)/length(train_targets);

%Estimate mean and covariance for class 1
m1 = mean(train_features(:,train_one)');
s1 = zeros(2);
s1(1,1) = var(train_features(1,train_one) - m1(1));
s1(2,2) = var(train_features(2,train_one) - m1(2));
P1 = length(train_one)/length(train_targets);

%Find decision region
param_struct.m0 = m0;
param_struct.m1 = m1;
param_struct.s0 = s0;
param_struct.s1 = s1;
param_struct.w0 = 1;
param_struct.w1 = 1;
param_struct.p0 = P0;

D		= decision_region(param_struct, region);
