function [features, targets] = ADDC(train_features, train_targets, Nmu, region, plot_on)

%Reduce the number of data points using the Agglomerative clustering algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	Nmu				- Maximum number of output data points
%	region			- Decision region vector (unused)
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features		- New features
%	targets			- New targets

if (nargin < 5),
    plot_on = 0;
end

if (Nmu == 1),
    %If one center is needed, it is simply the average of the data
    features = mean(train_features')';
    targets  = (sum(train_targets)/length(train_targets) > 0.5);
    break
end
    
[D,L]				= size(train_features);
min_percentage = 0.001; %Points with count less than this will be removed
min_number		= 5; 		%Points with count less than this will also be removed

%Initialize the mu's
K		= 0; %Number of centroids
mu		= zeros(D,Nmu);
count = zeros(1,Nmu);

for i = 1:L,
   data = train_features(:,i);
   
   if (K > 0),
      %Find closest centriod
      dist 				= sum((mu(:,1:K) - data * ones(1,K)).^2);
      [temp, min_d]  = min(dist);
      mu(:,min_d)		= mu(:,min_d) + (data - mu(:,min_d)) / (count(:,min_d) + 1);
      count(:,min_d) = count(:,min_d) + 1;
   end
   
   if (K < Nmu),
	   %Add new centroid
      K = K + 1;
      mu(:,K) = data;
   else
      %Merge redundant centroids
      closest_i1 = 0;
      closest_i2 = 0;
      dist		  = 1e100;
      for i1 = 1:K,
         for i2 = 1:K,
            if (i1 ~= i2),
               temp_dist = norm(mu(:,i1)-mu(:,i2));
               if (temp_dist < dist),
                  dist = temp_dist;
                  closest_i1 = i1;
                  closest_i2 = i2;
               end
            end
         end
      end
      if ((count(closest_i1) + count(closest_i2)) > 0),
         mu(:,closest_i1)  = (mu(:,closest_i1)*count(closest_i1) + mu(:,closest_i2)*count(closest_i2)) / ...
      						     (count(closest_i1) + count(closest_i2));
	      count(closest_i1) = count(closest_i1) + count(closest_i2);
   	   mu(:,closest_i2)  = data;
         count(closest_i2) = 0;
      end
   end
   
   if (plot_on == 1),
       plot_process(mu)
   end
   
end

%Post-processing
keep 		= find(count(1:K) > max(min_percentage*L,min_number));
features = mu(:,keep);
Nmu		= length(keep);

%Classify all the features to one of the mu's (1-NN)
dist = zeros(Nmu,L);
for i = 1:Nmu,
   dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
end
   
%Label the points
if (Nmu > 1),
   [m,label] = min(dist);
	targets = zeros(1,Nmu);
	for i = 1:Nmu,
   	if (length(train_targets(:,find(label == i))) > 0),
      	targets(i) = (sum(train_targets(:,find(label == i)))/length(train_targets(:,find(label == i))) > .5);
	   end
   end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end

