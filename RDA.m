function D = RDA (train_features, train_targets, lamda, region)

% Classify using the Regularized descriminant analysis (Friedman shrinkage algorithm)
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	lamda		- Parameter for the algorithm
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

train_one  = find(train_targets == 1);
train_zero = find(train_targets == 0);

%Estimate MLE mean and covariance for class 0
m0 = mean(train_features(:,train_zero)');
s0 = cov(train_features(:,train_zero)',1);
n0 = length(train_zero);

%Estimate MLE mean and covariance for class 1
m1 = mean(train_features(:,train_one)');
s1 = cov(train_features(:,train_one)',1);
n1 = length(train_one);

p0 = n0 / (n0+n1);

%Shrink for class 0
S      = n0 * s0;
n		 = n0;
sigma0 = zeros(2);
nk		 = n;
sk	    = S;
   
for i = 1:n,
   sk		 = (1 - lamda)*sk + lamda*S;
   nk		 = (1 - lamda)*nk + lamda*n;
   sigma0 = sk / nk;
   sigma0 = (1 - lamda) * sigma0 + lamda/2*trace(sigma0)*eye(2);
   sk		 = sigma0 * nk;
end
   
%Shrink for class 1
S      = n1 * s1;
n		 = n1;
sigma1 = zeros(2);
nk		 = n;
sk	    = S;
   
for i = 1:n,
   sk		 = (1 - lamda)*sk + lamda*S;
   nk		 = (1 - lamda)*nk + lamda*n;
   sigma1 = sk / nk;
   sigma1 = (1 - lamda) * sigma1 + lamda/2*trace(sigma1)*eye(2);
   sk     = sigma1 * nk;
end

param_struct.m0 = m0;
param_struct.m1 = m1;
param_struct.s0 = sigma0;
param_struct.s1 = sigma1;
param_struct.w0 = 1;
param_struct.w1 = 1;
param_struct.p0 = p0;

D	= decision_region(param_struct, region);

