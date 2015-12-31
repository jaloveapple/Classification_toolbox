function [features, train_targets, w] = MultipleDiscriminantAnalysis(train_features, train_targets)

%Reshape the data points using multiple descriminant analysis
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%
%Outputs
%	features			- New features
%	targets			- New targets
%  w					- Weights vector

[D, N]	= size(train_features);
Utargets	= unique(train_targets);
x			= train_features;

if (size(train_features,1) < length(Utargets)-1),
   error('Number of classes must be equal to or smaller than input dimension')
end

%Estimate Sw and Sb
m		= mean(x')';
Sw_i	= zeros(length(Utargets), D, D);
Sb_i	= zeros(length(Utargets), D, D);
for i = 1:length(Utargets),
   indices		= find(train_targets == Utargets(i)); 
   m_i			= mean(x(:,indices)')';
   Sw_i(i,:,:) = (x(:, indices) - m_i*ones(1,length(indices)))*(x(:, indices) - m_i*ones(1,length(indices)))';
   Sb_i(i,:,:) = length(indices)*(m_i - m)*(m_i - m)';
end
Sw		= squeeze(sum(Sw_i));
Sb		= squeeze(sum(Sb_i));

[v, d]= eig(Sb, Sw);

w			= v(:,1:length(Utargets)-1);
features = w'*x;