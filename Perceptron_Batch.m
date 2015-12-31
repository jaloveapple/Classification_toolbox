function D = Perceptron_Batch(train_features, train_targets, params, region)

% Classify using the batch Perceptron algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	param		- [Num iter, Theta (Convergence criterion), Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[c, r]		   = size(train_features);
[Max_iter, theta, eta] = process_params(params);

train_features = [train_features ; ones(1,r)];
train_one      = find(train_targets == 1);
train_zero     = find(train_targets == 0);

%Preprocessing
y              = train_features;
y(:,train_zero)= -y(:,train_zero);

%Initial weights
a              = sum(y')';
iter  	       = 0;

update		   = 10*theta;

while ((sum(abs(update)) > theta) & (iter < theta))
   iter = iter + 1;
   
   %Find all incorrectly classified samples, Yk
   Yk			= find(sign(a'*train_features.*(2*train_targets-1)) < 0);
   update	= eta * sum(y(:,Yk)')';
   
   % a <- a + eta*sum(Yk)
   a = a + update;
end

if (iter == theta),
   disp(['Maximum iteration (' num2str(theta) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D       = (a(1).*x + a(2).*y + a(c+1)> 0);
