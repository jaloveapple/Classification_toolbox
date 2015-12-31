function D = Perceptron(train_features, train_targets, alg_param, region)

% Classify using the Perceptron algorithm (Fixed increment single-sample perceptron)
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	alg_param	- Either: Number of iterations, weights vector or [weights, number of iterations]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[c, r]		   = size(train_features);

%Weighted Perceptron or not?
switch length(alg_param),
case r + 1,
    %Ada boost form
    p           = alg_param(1:end-1);
    max_iter    = alg_param(end);
case {r,0},
    %No parameter given
    p           = ones(1,r);
    max_iter    = 5000;
otherwise
    %Number of iterations given
    max_iter    = alg_param;
    p           = ones(1,r);
end

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
   indice = 1 + floor(rand(1)*n);
   if (a' *  y(:,indice) <= 0)
      a = a + p(indice)* y(:,indice);
   end
end

if (iter == max_iter)&(length(alg_param)~= r + 1),
   disp(['Maximum iteration (' num2str(max_iter) ') reached']);
end

%Find decision region
N		= region(5);
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);

D       = (a(1).*x + a(2).*y + a(c+1)> 0);
