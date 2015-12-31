function D = Minimum_Cost(train_features, train_targets, lambda, region)

% Classify using the minimum error criterion via histogram estimation of the densities
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	lambda  - Cost matrix
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace


train_one   = find(train_targets == 1);
train_zero  = find(train_targets == 0);
P0          = length(train_zero)/length(train_targets);
P1          = length(train_one)/length(train_targets);

Nbins       = max(3,floor(size(train_features,2).^(1/3)));
p0          = high_histogram(train_features(:,train_zero),Nbins,region(1:end-1));
p1          = high_histogram(train_features(:,train_one),Nbins,region(1:end-1));

decision    = (lambda(2,1) - lambda(1,1))*p0*P0 < (lambda(1,2) - lambda(2,2))*p1*P1;

%Make decision region
x           = linspace(region(1),region(2),region(5));
xx          = linspace(region(1),region(2),Nbins);
y           = linspace(region(3),region(4),region(5));
yy          = linspace(region(3),region(4),Nbins);

D           = interp2(xx, yy', decision, x, y')'>.5;