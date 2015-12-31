function D = Perceptron_VIM(train_features, train_targets, params, region)

% Classify using the variable incerement Perceptron with margin algorithm 
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	param		- [Num iter, Margin, Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[c, n]	 	    = size(train_features);
[theta, b, eta] = process_params(params);

train_features  = [train_features ; ones(1,n)];
train_zero      = find(train_targets == 0);

%Preprocessing
y               = train_features;
y(:,train_zero) = -y(:,train_zero);

%Initial weights
a               = sum(y')';

iter			= 0;
k				= 0;

while ((sum(a'*y <= b)>0) & (iter < theta))
   iter = iter + 1;
   
   %k <- (k+1) mod n
   k = mod(k+1,n);
   if (k == 0), 
      k = n;
   end
   
   if (a'*y(:,k) <= b),
	   % a <- a + eta*sum(Yk)
      a = a + eta * y(:,k);
   end
   
end

if (iter == theta),
   disp(['Maximum iteration (' num2str(theta) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D     = (a(1).*x + a(2).*y + a(c+1)> 0);
