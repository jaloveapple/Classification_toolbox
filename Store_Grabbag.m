function D = Store_Grabbag(train_features, train_targets, Knn, region)

% Classify using the store-grabbag algorithm (an improvement on the nearest neighbor)
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Knn		- Number of nearest neighbors
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

L		= length(train_features);
N		= region(5);
D		= zeros(N);

%Placing first sample in STORE
Store_features(:,1) = train_features(:,1);
Store_targets = train_targets(1);
Grabbag_targets = [];
Grabbag_features = [];

for i = 2:L,
   target = Knn_Rule(train_features(:,i), Store_features, Store_targets, Knn);
   if target == train_targets(i)
      Grabbag_features = [Grabbag_features , train_features(:,i)];  
      Grabbag_targets = [Grabbag_targets train_targets(i)];
   else
      Store_features = [Store_features, train_features(:,i)];
      Store_targets  = [Store_targets train_targets(i)];
   end 
end      

New_Grabbag_features = Grabbag_features;

while (Grabbag_features ~= New_Grabbag_features)
   Grabbag_features = New_Grabbag_features;
   New_Grabbag_targets = [];
   for i = 1:length(Grabbag_features),
      target = Knn_Rule(Grabbag_features(:,i), Store_features, Store_targets);
   	if target == train_targets(i)
      	New_Grabbag_features = [New_Grabbag_features, train_features(:,i)];  
      	New_Grabbag_targets  = [New_Grabbag_targets train_targets(i)];
   	else
      	Store_features = [Store_features, train_features(:,i)];
      	Store_targets  = [Store_targets , train_targets(i)];
      end
   end
end
    
      
disp(['Calling Nearest Neighbor algorithm']);
D = Nearest_Neighbor(Store_features, Store_targets, Knn, region);

%END

function target = Knn_Rule(Sample, Store_features, Store_targets, Knn)
%Classify a sample using the NN rule

for i = 1:length(Store_targets),
   %Find the k nearest neighbours
   dist(i) = sqrt((Sample(1)-Store_features(1,i)).^2+(Sample(2)-Store_features(2,i)).^2);  
end
[sorted_dist, indices] = sort(dist);

if length(Store_targets) <= Knn
   k_nearest = Store_targets;
else
   k_nearest = Store_targets(indices(1:Knn));
end

target = (sum(k_nearest) > Knn/2);

