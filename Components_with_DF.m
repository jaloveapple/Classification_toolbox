function D = Components_with_DF(train_features, train_targets, Ncomponents, region)

% Classify points using component classifiers with discriminant functions
% Inputs:
% 	train_features	- Train features
%	train_targets	- Train targets
%	Ncomponents		- Number of component classifiers
%	region			- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D					- Decision sufrace
%
% This implementation works with logistic component classifiers and a softmax gating function
% The parameters of the components are learned using Newton descent, and the parameters
% of the gating system using gradient descent

[Ndim, M] 	    = size(train_features);
Ndim			= Ndim + 1;
x				= [train_features; ones(1,M)];
y				= train_targets;
theta			= zeros(Ndim, Ncomponents)+eps;
alpha			= randn(Ndim, Ncomponents);
alpha			= sqrtm(cov(x',1)+randn(Ndim))*alpha + mean(x')'*ones(1,Ncomponents);

old_err		    = 10;
err			    = 1;

while ((err > 1/M) & (err < old_err)),
    old_err = err;
    
    %Perform gradient descent on the component classifiers
    w			= exp(alpha'*x)./(ones(Ncomponents,1)*sum(exp(alpha'*x)));
    for i = 1:Ncomponents,
        p					= exp(theta(:,i)'*x)./(1+exp(theta(:,i)'*x));
        W					= diag(p.*(1-p));
        delta_theta_i	= inv(x*W*x')*x*(y.*w(i,:) - p)';
        if ~isfinite(sum(delta_theta_i)),
            delta_theta_i = 0;
        end
        theta(:,i)		= theta(:,i) + delta_theta_i;
    end
    
    %Perform gradient descent on the gating parameters
    p				= zeros(Ncomponents, M);
    for i = 1:Ncomponents,
        p(i,:)			= exp(theta(:,i)'*x)./(1+exp(theta(:,i)'*x));
    end
    h               = w.*p./(ones(Ncomponents,1)*sum(w.*p));
    dalpha          = (x*(h - w)');
    alpha			= alpha + dalpha;
    
    w				= exp(alpha'*x)./(ones(Ncomponents,1)*sum(exp(alpha'*x)));
    Y				= sum(w.*p);
    err			= sum(y ~= (Y>.5))/M;
    
    disp(['Error is ' num2str(err)]) 
end

%Build decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:), ones(N^2,1)]';

y				= exp(theta'*flatxy)./(ones(Ncomponents,N^2) + exp(theta'*flatxy));
u				= exp(alpha'*flatxy)./(ones(Ncomponents,1)*sum(exp(alpha'*flatxy)));
D				= reshape(sum(y.*u)>.5,N,N);