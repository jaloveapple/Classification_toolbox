function [features, targets, label, W] = Leader_Follower(train_features, train_targets, params, region, plot_on)

%Reduce the number of data points using the basic leader-follower clustering algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	params 			- Algorithm parameters: [[Min distance to connect, Rate of convergence]]
%	region			- Decision region vector: [-x x -y y number_of_points]
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets
%	label				- The labels given for each of the original features
%   W               - Weights matrice

if (nargin < 5),
    plot_on = 0;
end
[theta, eta] = process_params(params);

[D,L]	= size(train_features);

%Preprocessing
features = [train_features ; ones(1,L)];
features = features ./ (ones(D+1,1) * sqrt(sum(features.^2)));

%w1 <- x
w       = features(:,1);

for i = 2:L,
    %Accept new pattern x
    x   = features(:,i);
    
    %j <- argmin||x-wj|| (Find nearest cluster)
    dist  = sqrt(sum((w - x*ones(1,size(w,2))).^2));
    j     = find(min(dist) == dist);
    
    %if ||x-wj|| < theta
    if dist(j) < theta,
        %wj <- wj + eta*x
        w(:,j) = w(:,j) + eta*x;
    else
        %Add new w <- x
        w(:,end+1) = x;
    end
    
    w   = w ./ (ones(D+1,1) * sqrt(sum(w.^2)));
    
    if (plot_on == 1),
        %Assign each of the features to a center
        dist        = w'*features;
        [m, label]  = max(dist);
        centers     = zeros(D,size(w,2));
        for i = 1:size(w,2),
            in = find(label == i);
            if ~isempty(in)
                centers(:,i) = mean(train_features(1:2,find(label==i))')';
            else
                centers(:,i) = nan;
            end
        end
        plot_process(centers)
    else
        disp(['There are ' num2str(size(w,2)) ' clusters so far'])
    end
    
end 
       
%Assign each of the features to a center
N           = size(w,2);
dist        = w'*features;
[m, label]  = max(dist);
features    = zeros(D,N);
for i = 1:N,
    in = find(label == i);
    if ~isempty(in)
        features(:,i) = mean(train_features(1:2,find(label==i))')';
    else
        features(:,i) = nan;
    end
end

targets = zeros(1,N);
if (N > 1),
	for i = 1:N,
   	if (length(train_targets(find(label == i))) > 0),
      	targets(i) = (sum(train_targets(find(label == i)))/length(train_targets(find(label == i))) > .5);
   	end
	end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end
