function D = None(train_features, train_targets, Params, region)

% Make no classifications (Dummy function)
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Dummy		- Unused
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

D = zeros(region(length(region)));
