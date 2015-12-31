function D = NearestNeighborEditing(train_features, train_targets, params, region)

% Classify points using the nearest neighbor editing algorithm
% Inputs:
% 	train_features	- Train features
%	train_targets	- Train targets
%	Unused	- Unused
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

%Construct the Voronoi region of the data
D	= voronoi_regions(train_features,region);

mark	= zeros(1,size(train_features,2));
for i = 1:size(train_features,2),
   %For each prototype Xj, find the Voronoi neighbors of Xj
   [x,y] = find(D==i);
   
   if ~isempty(x),
      %x and y are the locations of the Voronoi region for the i-th prototype
      %These can be used to find the Voronoi neighbors
      around = [x-1 x+1 x x; y y y-1 y+1];   
      indices= find((around(:,1)>0) & (around(:,2)<=region(5)) & (around(:,2)>0) & (around(:,2)<=region(5)));
      around = around(indices,:);
      
      neighbors = zeros(1,size(around,1));
      for j = 1:length(neighbors),
         neighbors(j) = D(around(j,1),around(j,2));
      end
      neighbors = unique(neighbors);
      
      %If any neighbor is not from the same class, mark the i-th prototype
      if (length(unique(train_targets(neighbors))) > 1),
         mark(i) = 1;
      end
   end   
end

%Discard all unmarked points
prototypes	= find(mark == 1);
if isempty(prototypes)
   error('No prototypes found')
else
   D				= nearest_neighbor(train_features(:,prototypes),train_targets(prototypes),1,region);
end