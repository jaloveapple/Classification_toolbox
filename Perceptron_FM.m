function [D, a] = Perceptron_FM(train_features, train_targets, params, region)

% Classify using the Perceptron algorithm but at each iteration updating the worst-classified sample
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	params   	- [Maximum number of iterations, slack]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[max_iter, slack] = process_params(params);
rate	            = 0.1;

[c, r]          = size(train_features);
xi  			= ones(1,r)/r*slack;

train_features = [train_features ; ones(1,r)];
train_zero     = find(train_targets == 0);

%Preprocessing
y = train_features;
y(:,train_zero)= -y(:,train_zero);

%Initial weights
a              = sum(y')';
n			   = length(train_targets);
iter		   = 0;

while ((sum(sign(a'*train_features.*(2*train_targets-1))<0)>0) & (iter < max_iter))
   iter = iter + 1;
   %Find worst-classified sample
   A            = a'*train_features.*(2*train_targets-1)+xi;
   [m, indice]  = min(A);
   if (a' *  y(:,indice) <= 0)
      a = a + y(:,indice);
   end
   
   %Calculate the new slack vector
   xi(indice)   = xi(indice) + rate;
   xi	        = xi / sum(xi) * slack;
   
end

if (iter == max_iter),
   disp(['Maximum iteration (' num2str(max_iter) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D       = (a(1).*x + a(2).*y + a(c+1)> 0);
a       = a';