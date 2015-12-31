function D = Nearest_Neighbor(train_features, train_targets, Knn, region)

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Knn		- Number of nearest neighbors 
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

L			= length(train_targets);
N			= region(5);
x			= linspace (region(1),region(2),N);
y			= linspace (region(3),region(4),N);

D			= zeros(N);

if (L < Knn),
   error('You specified more neighbors than there are points.')
end

y_dist	= (ones(N,1) * train_features(2,:) - y'*ones(1,L)).^2;

for i = 1:N,
	if (i/50 == floor(i/50)),
      disp(['Finished ' num2str(i) ' lines out of ' num2str(N) ' lines.'])
   end

   x_dist = ones(N,1)  * (train_features(1,:)-x(i)).^2;
  	dist   = abs(x_dist + y_dist);   
   [sorted_dist, indices] = sort(dist');
   k_nearest = train_targets(indices(1:Knn,:));
   if (Knn > 1),
      D(:,i)    = (sum(k_nearest) > Knn/2)';  
   else
      D(:,i)	 = (k_nearest > 0)';
   end
   
end


