function [mu, sigma] = Bayesian_parameter_est(train_features, train_targets, sigma, region)

% Estimate the mean using the Bayesian parameter estimation for Gaussian mixture algorithm
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	sigma		- The covariance matrix for each class 
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	mu			- The estimated mean
%	sigma		- The estimated covariances

[N,M]		= size(train_features);
Uclasses = unique(train_targets);
Nuc		= length(Uclasses);

%Find initial estimates for mu and sigma for the classes
mu0		= zeros(Nuc, N);
sigma0	= zeros(Nuc, N, N);
for i = 1:Nuc,
	indices	= find(train_targets == Uclasses(i));
   mu0(i,:) = mean(train_features(:,indices)');
   sigma0(i,:,:) = sqrtm(cov(train_features(:,indices)',1));
end

%Now gradually find the Bayesian estimate
mu_n		= zeros(Nuc, N);
sigma_n	= zeros(Nuc, N, N);

%For each class, estimate the mean and variance for each class
for i = 1:Nuc,
	indices	= find(train_targets == Uclasses(i));
   for n = 2:length(indices),
      indices		= indices(randperm(length(indices)));
      samples		= indices(1:n);
      
      mu_n_hat		= 1/n*sum(train_features(:,samples)')';
      mu_n(i,:)	= (squeeze(sigma0(i,:,:))*inv(squeeze(sigma0(i,:,:))+1/n*squeeze(sigma(i,:,:)))*mu_n_hat + ...
                  + 1/n*squeeze(sigma(i,:,:))*inv(squeeze(sigma0(i,:,:))+1/n*squeeze(sigma(i,:,:)))*mu0(i,:)')';
      sigma_n(i,:,:) = squeeze(sigma0(i,:,:))*inv(squeeze(sigma0(i,:,:))+1/n*squeeze(sigma(i,:,:)))*1/n*squeeze(sigma(i,:,:));
   end
end

%Return the estimates
mu 	= mu_n;
sigma = sigma + sigma_n;
