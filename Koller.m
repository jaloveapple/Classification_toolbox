function [features, targets, remaining_features] = Koller(features, targets, Nfeatures, region)

% Koller and Sawami algorithm for feature selection
%
%	train_features	- Input features
%	train_targets	- Input targets
%	Nfeatures  		- Output dimension
%	region			- Decision region vector: [-x x -y y number_of_points]
%
%Outputs
%	features		- New features
%	targets			- New targets
%	remaining_features	- The numbers of the selected features

Kdist			= 2;					%How many features to group together

%First, calculate the cross-entropy matrix
gamma			= infomat(features,targets);
Nf              = length(gamma);

gamma           = gamma + abs(min(min(gamma)));
diagonal		= diag(gamma);
gamma			= gamma - diag(diagonal);
discarded   = [];

%Discard redundant features
gamma       = gamma';
for k = 1:Nf-Nfeatures,
   tic
   sgamma        = sort(gamma);
   sums			 = sum(sgamma(k:k+Kdist-1,:));
   sums(discarded) = inf;
   [m, min_i]	 = min(sums);
   discarded	 = [discarded min_i(1)];
   gamma(min_i(1),:) = 0;

   t = toc;
   disp([num2str(k) ':Discarded feature number: ' num2str(min_i(1)) '. This took ' num2str(t) '[sec]'])
end

remaining_features = 1:Nf;
remaining_features(discarded) = 0;
remaining_features = remaining_features(find(remaining_features~=0));

disp(['Last two remaining feature numbers: ' num2str(remaining_features)])

features = features(remaining_features, :);