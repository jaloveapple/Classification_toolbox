function [D, a_plus, a_minus] = Balanced_Winnow(train_features, train_targets, params, region)

% Classify using the balanced Winnow algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	params		- [Num iter, Alpha, Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%  a_plus	    - The positive weight vector
%  a_minus	    - The negative weight vector

[c, r]		    		  = size(train_features);
[Max_iter, alpha, eta] = process_params(params);
y 			              = [train_features ; ones(1,r)];
z        			     = train_targets;

%Initial weights
a_plus	        = sum(y')';
a_minus	        = -sum(y')';
iter  	        = 0;

while (iter < Max_iter)
   iter = iter + 1;
   
   for k = 1:r,
      if (sign(a_plus'*y(:,k) - a_minus'*y(:,k)) ~= sign(z(k)-.5)),
         if (z(k) == 1),
            a_plus	= alpha.^y(:,k).*a_plus;
            a_minus	= alpha.^-y(:,k).*a_minus;
         else
            a_plus	= alpha.^-y(:,k).*a_plus;
            a_minus	= alpha.^y(:,k).*a_minus;
         end
      end
   end
   
end

a = (a_plus + a_minus)/2;

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D       = (a(1).*x + a(2).*y + a(c+1)< 0.5);
