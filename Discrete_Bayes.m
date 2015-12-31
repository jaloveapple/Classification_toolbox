function D = Discrete_Bayes(train_features, train_targets, cost, region, test_feature)

% Classify discrete features using the Bayes decision theory
% Inputs:
% 	features		- Train features
%	targets			- Train targets
%	cost			- Cost for class 0 (Optional, Unused yet)
%	region			- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			    - Decision sufrace

%First, find out if the features are indeed discrete 
%Of course, since this program is in Matlab, we define discrete as having 
%no more than one decimal place 
if (sum(sum(train_features*10~=floor(train_features*10))) ~= 0),
   error('Features are not discrete (See the definition of discrete in the m-file)')
end

p0 = length(find(train_targets==0))/length(train_targets);

%Find how the features are distributed
N	= 0;
unique_features = [];
counts			 = [];
for i = 1:size(train_features,2),
   data    = train_features(:,i);
   indices = find(sum((data*ones(1,size(train_features,2))-train_features).^2)==0);
   if isempty(unique_features),
      unique_features(:,1) = data;
      counts(1,1)				= length(find(train_targets(indices) == 0));
      counts(2,1)				= length(find(train_targets(indices) == 1));
      N							= 2;
   else
      if isempty(find(sum((data*ones(1,size(unique_features,2))-unique_features).^2)==0)),
         %Add this feature to the bank
	      unique_features(:,N) = data;
   	   counts(1,N)				= length(find(train_targets(indices) == 0));
      	counts(2,N)				= length(find(train_targets(indices) == 1));
	      N							= N + 1;
      end
   end
end

Px_given_w = (counts ./ (ones(2,1)* sum(counts)));

Pw_given_x = Px_given_w .* ([p0;1-p0]*ones(1,N-1));
Pw_given_x = Pw_given_x ./ (ones(2,1)*sum(Pw_given_x));

unique_targets = Pw_given_x(2,:) > Pw_given_x(1,:);

%Interpolate this over the whole decision region
%Originally, this isn't needed, but it is done because of the structure of the toolbox
D = nearest_neighbor(unique_features,unique_targets,1,region);
