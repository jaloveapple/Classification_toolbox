function [features, targets] = LVQ1(train_features, train_targets, Nmu, region, plot_on)

%Reduce the number of data points using linear vector quantization
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
%OR
%	D					- Decision region

if (nargin < 5),
    plot_on = 0;
end

alpha = 0.9;
L		= length(train_targets);
dist	= zeros(Nmu,L);
label = zeros(1,L);

%Initialize the mu's
mu			= randn(2,Nmu);
mu			= sqrtm(cov(train_features',1))*mu + mean(train_features')'*ones(1,Nmu);
mu_target= rand(1,Nmu)>0.5;
old_mu	= zeros(2,Nmu);

while (sum(sum(abs(mu - old_mu))) > 0.1),
   old_mu = mu;
   
   %Classify all the features to one of the mu's
   for i = 1:Nmu,
      dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
   end
   
   %Label the points
   [m,label] = min(dist);
   
   %Label the mu's
	for i = 1:Nmu,
   	if (length(train_targets(:,find(label == i))) > 0),
      	mu_target(i) = (sum(train_targets(:,find(label == i)))/length(train_targets(:,find(label == i))) > .5);
	   end
	end	
   
   %Recompute the mu's
   for i = 1:Nmu,
      indices = find(label == i);
      if ~isempty(indices),
         Q		  = ones(2,1) * (2*(train_targets(indices) == mu_target(i)) - 1);
         mu(:,i) = mu(:,i) + mean(((train_features(:,indices)-mu(:,i)*ones(1,length(indices))).*Q)')'*alpha;
      end
      
   end
   
   alpha = 0.95 * alpha;
   
   if (plot_on == 1),
       plot_process(mu)
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

if (nargout == 1),
   features = Nearest_Neighbor(features, targets, 1, region);
end
