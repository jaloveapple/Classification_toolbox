function [D, a] = Relaxation_BM(train_features, train_targets, params, region)

% Classify using the batch relaxation with margin algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	param		- [Max iter, Margin, Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   a          - Classifier weights

[c, n]				    = size(train_features);
[Max_iter, b, eta]	 = process_params(params);

y               = [train_features ; ones(1,n)];
train_zero      = find(train_targets == 0);

%Preprocessing
processed_features = y;
processed_features(:,train_zero) = -processed_features(:,train_zero);

%Initial weights
a               = sum(processed_features')';
iter  	        = 0;
Yk				= [1];

while (~isempty(Yk) & (iter < Max_iter))
   iter = iter + 1;
   
   %If a'y_j <= b then append y_j to Yk
   Yk = [];
   for k = 1:n,
   	if (a'*processed_features(:,k) <= b),
         Yk = [Yk k];
      end
   end
   
   if isempty(Yk),
      break
   end
   
   % a <- a + eta*sum((b-w'*Yk)/||Yk||*Yk)
   grad			= (b-a'*y(:,Yk))./sum(y(:,Yk).^2);
   update		= sum(((ones(c+1,1)*grad).*y(:,Yk))')';
   a            = a + eta * update;
end

if (iter == Max_iter),
   disp(['Maximum iteration (' num2str(Max_iter) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D       = (a(1).*x + a(2).*y + a(c+1)> 0);
a       = a';