function [features, targets] = AGHC(train_features, train_targets, params, region, plot_on)

%Reduce the number of data points using the agglomerative hierarchical clustering algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	params			- Parameters: [Number of output data points, distance type]
%						  Distance can be min, max, avg, or mean
%	region			- Decision region vector: [-x x -y y number_of_points]
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets

if (nargin < 5),
    plot_on = 0;
end

[c, method] = process_params(params);
[D,c_hat]	= size(train_features);
label       = 1:c_hat;
n           = ones(1,c_hat);

%Compute distances
N           = size(train_features,2);
temp        = repmat(train_features,[1 1 N]);
dist        = sqrt(squeeze(sum((temp - permute(temp, [1 3 2])).^2)));

while (c_hat > c),
    Uc       = unique(label);
    Nc       = length(Uc);
    new_dist = zeros(Nc);

    switch method
    case 'min'
        %Find minimum distance between vectors from different clusters
        
        %For each two clusters, find the shortest distance between vectors
        for i = 1:Nc,
            i_in = find(label == Uc(i));
            for j = 1:Nc,
                j_in = find(label == Uc(j));
                new_dist(i,j) = min(min(dist(i_in,j_in)));
            end
        end
        new_dist    = new_dist + eye(Nc)*1e33;
        [i,j]   = find(new_dist == min(min(new_dist)));
        i = Uc(i(1)); j = Uc(j(1));
    case 'max'
        %Find maximum distance between vectors from different clusters
        
        %For each two clusters, find the longest distance between vectors
        for i = 1:Nc,
            i_in = find(label == Uc(i));
            for j = 1:Nc,
                j_in = find(label == Uc(j));
                new_dist(i,j) = max(max(dist(i_in,j_in)));
            end
        end
        new_dist = new_dist .* (ones(Nc)-eye(Nc));
        [i,j]   = find(new_dist == max(max(new_dist)));
        i = Uc(i(1)); j = Uc(j(1));
        
    case 'avg'
        %Find average distance between vectors from different clusters
        
        %For each two clusters, find the average distance between vectors in one cluster to each vector in the other cluster
        for i = 1:Nc,
            i_in = find(label == Uc(i));
            for j = 1:Nc,
                j_in = find(label == Uc(j));
                new_dist(i,j) = mean(mean(dist(i_in,j_in)))/(length(j_in)*length(i_in));
            end
        end
        new_dist = new_dist .* (ones(Nc)-eye(Nc));
        [i,j]   = find(new_dist == max(max(new_dist)));
        i = Uc(i(1)); j = Uc(j(1));
        
    case 'mean'
        %Find meanimum distance between cluster centers 
        
        %For each two clusters, find the average distance between vectors in one cluster to each vector in the other cluster
        for i = 1:Nc,
            i_in = find(label == Uc(i));
            for j = 1:Nc,
                j_in = find(label == Uc(j));
                new_dist(i,j) = sum((mean(train_features(:,i_in)')'-mean(train_features(:,j_in)')').^2);
            end
        end
        new_dist    = new_dist + eye(Nc)*1e33;
        [i,j]   = find(new_dist == min(min(new_dist)));
        i = Uc(i(1)); j = Uc(j(1));
    otherwise
        error('Distance method unknown')
    end
      
    %Merge cluster i with cluster j
    label(find(label == j)) = i;
    
    c_hat = c_hat - 1;
    
    %Computer cluster centers
    Uc       = unique(label);
    Nc       = length(Uc);
    features = zeros(D,Nc);
    for i = 1:Nc,
        features(:,i) = mean(train_features(:,find(label == Uc(i)))')';
    end
    
    if (plot_on == 1),
        plot_process(features)
    end

end
 
%Make the decision region
targets = zeros(1,c);
Uc      = unique(label);
if (c > 1),
	for i = 1:c,
       	if (length(train_targets(:,find(label == Uc(i)))) > 0),
          	targets(i) = (sum(train_targets(:,find(label == Uc(i))))/length(train_targets(:,find(label == Uc(i)))) > .5);
       	end
	end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end
