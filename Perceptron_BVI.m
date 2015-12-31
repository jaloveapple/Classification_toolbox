function D = Perceptron_BVI(train_features, train_targets, params, region)

% Classify using the batch variable increment Perceptron algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	param		- [Num iter, Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[c, n]		    = size(train_features);
[theta, eta]	 = process_params(params);

train_features  = [train_features ; ones(1,n)];
train_zero      = find(train_targets == 0);

%Preprocessing
y               = train_features;
y(:,train_zero) = -y(:,train_zero);
a               = sum(y')';

%Initial weights
iter  	        = 0;
Yk				= [1];

while (~isempty(Yk) & (iter < theta))
   iter = iter + 1;
   
   %If y_j is misclassified then append y_j to Yk
   Yk = [];
   for k = 1:n,
      if (sign(a'*train_features(:,k).*(2*train_targets(:,k)-1)) < 0),
         Yk = [Yk k];
      end
   end
   
   % a <- a + eta*sum(Yk)
   a = a + eta * sum(y(:,Yk)')';
end

if (iter == theta),
   disp(['Maximum iteration (' num2str(theta) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D     = (a(1).*x + a(2).*y + a(c+1)> 0);
