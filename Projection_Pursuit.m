function [D, V, Wo] = Projection_Pursuit(train_features, train_targets, Ncomponents, region)

% Classify using projection pursuit regression
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	Ncomponents - Number of components to project on
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%  V        - Components weights
%  Wo       - Output unit weights

[Ni, M] = size(train_features);

iter			= 0;
NiterDisp	= 1;
maxIter 		= 50;

train_targets = (train_targets>0)*2-1;
train_features= [train_features; ones(1,M)];
Ni		  = Ni + 1;
No		  = 1; %Number of output units

V		  = rand(Ni, Ncomponents);
Wo		  = rand(No, Ncomponents+1);
gradJ	  = 1;
J		  = 1;

%Find the regression parameters
while ((gradJ > 1e-2) & (iter < maxIter)),
   iter = iter + 1;
   
   %Optimize for the components
   J1 = inline('sum((t - Wo*[tanh(V''*x); ones(No, M)]).^2)','V','t','Wo','x','No','M');
	V  = fminunc(J1, V, [], train_targets, Wo, train_features, No, M);       
   
   %Optimize the weights
   J2 = inline('sum((t - Wo*[tanh(V''*x); ones(No, M)]).^2)','Wo','t','V','x','No','M');
	Wo = fminunc(J2, Wo, [], train_targets, V, train_features, No, M);       
   
   %Evaluate the error
   oldJ	= J;
   J		= feval(J1, V, train_targets, Wo, train_features, No, M);
   gradJ = abs(oldJ - J);
end

if (iter == maxIter),
   disp('Optimization terminated after reaching the maximum iteration.')
else
   disp(['Converged after ' num2str(iter) ' iterations.'])
end

%Build a decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:), ones(N^2,1)]';
PPR			= Wo*[tanh(V'*flatxy); ones(No, N^2)];
D				= reshape(PPR, N, N)> 0;
