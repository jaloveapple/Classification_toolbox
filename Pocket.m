function [D, w_pocket] = Pocket(train_features, train_targets, alg_param, region)

% Classify using the pocket algorithm (an improvement on the perceptron)
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	alg_param	- Either: Number of iterations, weights vector or [weights, number of iterations]
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%	w			- Decision surface parameters
[c, r] = size(train_features);

%Weighted Pocket or not?
switch length(alg_param),
case r + 1,
    %Ada boost form
    p           = alg_param(1:end-1);
    max_iter    = alg_param(end);
case {r, 0},
    %No parameter given
    p           = ones(1,r);
    max_iter    = 500;
otherwise
    %Number of iterations given
    max_iter    = alg_param;
    p           = ones(1,r);
end

train_features = [train_features ; ones(1,r)];
train_one      = find(train_targets == 1);
train_zero     = find(train_targets == 0);

%Preprocessing
processed_features = train_features;
processed_features(:,train_zero) = -processed_features(:,train_zero);

%Initial weights
w_percept   = sum(processed_features')';
%w_percept   = train_features .* (ones(c+1,1) * (2*(train_targets-0.5)));
%w_percept	= rand(c+1,1);
w_pocket	= rand(c+1,1);

correct_classified = 0;
n						 = length(train_targets);
iter					 = 0;

while ((longest_run(w_percept, processed_features) < n) & (iter < max_iter))
   iter = iter + 1;
   %Every 10 points, do the pocket switchover
   for i = 1:10,
      indice = 1 + floor(rand(1)*n);
      if (w_percept' * processed_features(:,indice) <= 0)
         w_percept = w_percept + p(indice) * processed_features(:,indice);
      end
   end
   %Find if it is neccessary to change weights:
   if (longest_run(w_percept, processed_features) > longest_run(w_pocket, processed_features)),
      w_pocket = w_percept;
   end
end

if (iter == max_iter)&(length(alg_param)~= r + 1),
   disp(['Maximum iteration (' num2str(max_iter) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D     = (w_pocket(1).*x + w_pocket(2).*y + w_pocket(c+1)> 0);
w_pocket = w_pocket';



function L = longest_run(weights, features)

%Find the length of the longest run of correctly classified random points
n           = length(features);
indices     = randperm(n);
L           = 0;
correct     = 1;

for i = 1:n,
   if (weights' * features(:,indices(i)) <= 0)	%Find if it is correctly classified
      break
   end
   L = i;
end
