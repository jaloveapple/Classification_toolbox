function [features, targets, label] = k_means(train_features, train_targets, Nmu, region, plot_on)

%Reduce the number of data points using the k-means algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	Nmu				- Number of output data points
%	region			- Decision region vector: [-x x -y y number_of_points]
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets
%	label				- The labels given for each of the original features

if (nargin < 5),
    plot_on = 0;
end

[D,L]	= size(train_features);
dist	= zeros(Nmu,L);
label = zeros(1,L);

%Initialize the mu's
mu			= randn(D,Nmu);
mu			= sqrtm(cov(train_features',1))*mu + mean(train_features')'*ones(1,Nmu);
old_mu	= zeros(D,Nmu);

switch Nmu,
case 0,
    mu      = [];
    label   = [];
case 1,
   mu		= mean(train_features')';
   label	= ones(1,L);
otherwise
   while (sum(sum(mu == old_mu)) == 0),
      old_mu = mu;
      
      %Classify all the features to one of the mu's
      for i = 1:Nmu,
         dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
      end
      
      %Label the points
      [m,label] = min(dist);
      
      %Recompute the mu's
      for i = 1:Nmu,
         mu(:,i) = mean(train_features(:,find(label == i))')';
      end

      if (plot_on == 1),
        plot_process(mu)
      end

   end
end
   
%Make the decision region
targets = zeros(1,Nmu);
if (Nmu > 1),
	for i = 1:Nmu,
   	if (length(train_targets(:,find(label == i))) > 0),
      	targets(i) = (sum(train_targets(:,find(label == i)))/length(train_targets(:,find(label == i))) > .5);
   	end
	end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end

features = mu;