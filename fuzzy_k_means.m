function [features, targets] = fuzzy_k_means(train_features, train_targets, Nmu, region, plot_on)

%Reduce the number of data points using the fuzzy k-means algorithm
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

if (nargin < 5),
    plot_on = 0;
end

b		= 2;
L		= length(train_targets);
dist	= zeros(Nmu,L);
label = zeros(1,L);

%Initialize the mu's
mu			= randn(2,Nmu);
mu			= sqrtm(cov(train_features',1))*mu + mean(train_features')'*ones(1,Nmu);
old_mu	= zeros(2,Nmu);

%Initialize the P's
P		= randn(Nmu,L);
old_P	= zeros(Nmu,L);

while ((sum(sum(mu == old_mu)) == 0) & (sum(sum(P == old_P)) == 0)),
   old_mu = mu;
   old_P  = P;
   
   %Classify all the features to one of the mu's
   for i = 1:Nmu,
      dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
   end
   
   %Recompute P's
   for i = 1:Nmu,
      P(i,:) = (1./dist(i,:)).^(1/(b-1));
   end
   P = P ./ (ones(Nmu,1) * sum(P));
   
   %Recompute the mu's
   for i = 1:Nmu,
      mu(:,i) = (sum((((ones(2,1)*P(i,:)).^b).*train_features)')./sum(((ones(2,1)*P(i,:)).^b)'))';
   end

   if (plot_on == 1),
      plot_process(mu)
   end

end

%Make the decision region
[m,label] = max(P);
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