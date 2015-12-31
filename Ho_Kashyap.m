function [D, w_percept, b] = Ho_Kashyap(train_features, train_targets, params, region)

% Classify using the using the Ho-Kashyap algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	params		- [Type(Basic/Modified), Maximum iteration, Convergence criterion, Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   w_percept   - Classifier weights
%   b           - Margin

[c, n]		   = size(train_features);

[type, Max_iter, b_min, eta] = process_params(params);

train_features  = [train_features ; ones(1,n)];
train_zero      = find(train_targets == 0);

%Preprocessing (Needed so that b>0 for all features)
processed_features = train_features;
processed_features(:,train_zero) = -processed_features(:,train_zero);
processed_targets  = 2*train_targets-1;

b                  = ones(1,n);
Y                  = processed_features;
a                 = pinv(Y')*b';
k	               = 0;
e    	           = 1e3;
found              = 0;

while ((sum(abs(e) > b_min)>0) & (k < Max_iter) &(~found))

    %k <- (k+1) mod n
    k = k+1;

    %e <- Ya - b
    e       = (Y' * a)' - b;
   
    %e_plus <- 1/2(e+abs(e))
    e_plus  = 0.5*(e + abs(e));
   
    %b <- b + 2*eta*e_plus
    b       = b + 2*eta*e_plus;
    
    if strcmp(type,'Basic'),
        %a <- pinv(Y)*b;   
        a = pinv(Y')*b';
    else
        %a <- a + eta*pinv(Y)*|e_plus|;   
        a = a + eta*pinv(Y')*e_plus';
    end        
    
end

if (k == Max_iter),
   disp(['No solution found']);
else
   disp(['Did ' num2str(k) ' iterations'])
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D       = (a(1).*x + a(2).*y + a(c+1)> 0);
a       = a';
