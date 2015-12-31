function [D, a] = Relaxation_SSM(train_features, train_targets, params, region)

% Classify using the single-sample relaxation with margin algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	param		- [Max iter, Margin, Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   a          - Classifier weights

[c, n]		    = size(train_features);
[Max_iter, b, eta]	 = process_params(params);

y               = [train_features ; ones(1,n)];
train_zero      = find(train_targets == 0);

%Preprocessing
processed_features = y;
processed_features(:,train_zero) = -processed_features(:,train_zero);

%Initial weights
a               = sum(processed_features')';
iter  	        = 0;
k				= 0;

while ((sum(a'*processed_features < b)>0) & (iter < Max_iter))
    iter = iter + 1;
    
    %k <- (k+1) mod n
    k = mod(k+1,n);
    if (k == 0), 
        k = n;
    end
    
    if (a'*processed_features(:,k) <= b),
        % a <- a + eta*sum((b-w'*Yk)/||Yk||*Yk)
        grad			= (b-a'*y(:,k))./sum(y(:,k).^2);
        update		= grad.*y(:,k);
        a 	= a + eta * update;
    end
    
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