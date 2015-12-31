function D = LMS(train_features, train_targets, params, region)

% Classify using the least means square algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	param		- [Maximum iteration Theta (Convergence criterion), Convergence rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[c, n]          			= size(train_features);
[Max_iter, theta, eta]	= process_params(params);

y               = [train_features ; ones(1,n)];
train_zero      = find(train_targets == 0);

%Preprocessing
processed_features = y;
processed_features(:,train_zero) = -processed_features(:,train_zero);
b               = 2*train_targets - 1; 

%Initial weights
a               = sum(processed_features')';
iter  	        = 0;
k				= 0;

update	        = 1e3;

while ((sum(abs(update)) > theta) & (iter < Max_iter))
    iter = iter + 1;
    
    %k <- (k+1) mod n
    k = mod(k+1,n);
    if (k == 0), 
        k = n;
    end
    
    % a <- a + eta*(b-a'*Yk)*Yk'
    update	= eta*(b(k) - a'*y(:,k))'*y(:,k);
    a	    = a + update;
    
end

if (iter == Max_iter),
    disp(['Maximum iteration (' num2str(Max_iter) ') reached']);
else
    disp(['Did ' num2str(iter) ' iterations'])
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D     = (a(1).*x + a(2).*y + a(c+1)> 0);
