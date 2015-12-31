function [features, targets] = LVQ3(train_features, train_targets, Nmu, region, plot_on)

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

if ((sum(train_targets) == length(train_targets)) | (sum(~train_targets) == length(train_targets))),
    error('LVQ3 works only if there are features from both classes.')
end

L		= length(train_targets);
alpha = 10/L;
[D,r] = size(train_features);
dist	= zeros(Nmu,L);
label = zeros(1,L);

window = 0.25;
epsilon= 0.25;

%Initialize the mu's
mu			= randn(D,Nmu);
mu			= sqrtm(cov(train_features',1))*mu + mean(train_features')'*ones(1,Nmu);
mu_target= [zeros(1,floor(Nmu/2)) ones(1,Nmu-floor(Nmu/2))];
old_mu	= zeros(D,Nmu);

iterations = 0;

while ((sum(sum(abs(mu - old_mu))) > 0.01) & (iterations < 1e4)),
   iterations = iterations + 1;
   old_mu = mu;
   
   %Classify all the features to one of the mu's
   for i = 1:Nmu,
      dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
   end
   
   %Label the points
   [dist,label] = sort(dist);
   closest		 = dist(1:2,:);
    
   %Compute windows
   in_window = (min(closest(1,:)./closest(2,:), closest(2,:)./closest(1,:)) > (1-window)/(1+window));
   indices	 = find(in_window);
   
   %Move the mu's
   for i = 1:length(indices),
      x	 = indices(i);
      mu1 = label(1,x);
      mu2 = label(2,x);
      if ((train_targets(x) == mu_target(mu1)) & (train_targets(x) == mu_target(mu2))),
         mu(:,mu1) = mu(:,mu1) + epsilon * alpha * (train_features(:,x) - mu(:,mu1));
         mu(:,mu2) = mu(:,mu2) + epsilon * alpha * (train_features(:,x) - mu(:,mu2));
      else
         if (train_targets(x) == mu_target(mu1)),
	         mu(:,mu1) = mu(:,mu1) + alpha * (train_features(:,x) - mu(:,mu1));
            mu(:,mu2) = mu(:,mu2) - alpha * (train_features(:,x) - mu(:,mu2));
         else
	         mu(:,mu1) = mu(:,mu1) - alpha * (train_features(:,x) - mu(:,mu1));
            mu(:,mu2) = mu(:,mu2) + alpha * (train_features(:,x) - mu(:,mu2));
         end
      end
   end

   alpha = 0.95 * alpha;

   if (plot_on == 1),
       plot_process(mu)
   end

end

%Make the decision region
targets  = mu_target;
features = mu;

if (nargout == 1),
   features = Nearest_Neighbor(features, targets, 1, region);
end
