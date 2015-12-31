function [D, a_star] = SVM(train_features, train_targets, params, region)

% Classify using (a very simple implementation of) the support vector machine algorithm
% 
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params	- [kernel, kernel parameter, solver type, Slack]
%               Kernel can be one of: Gauss, RBF (Same as Gauss), Poly, Sigmoid, or Linear
%               The kernel parameters are:
%                   RBF kernel  - Gaussian width (One parameter)
%                   Poly kernel - Polynomial degree
%                   Sigmoid     - The slope and constant of the sigmoid (in the format [1 2], with no separating commas)
%					Linear		- None needed
%               Solver type can be one of: Perceptron, Quadprog
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%	a			- SVM coeficients
%
% Note: The number of support vectors found will usually be larger than is actually 
% needed because both solvers are approximate.

[Dim, Nf]       = size(train_features);
Dim             = Dim + 1;
train_features(Dim,:) = ones(1,Nf);
z					 = 2*(train_targets>0) - 1; 

%Get kernel parameters
[kernel, ker_param, solver, slack] = process_params(params);

%Transform the input features
y	= zeros(Nf);
switch kernel,
case {'Gauss','RBF'},
   for i = 1:Nf,
      y(:,i)    = exp(-sum((train_features-train_features(:,i)*ones(1,Nf)).^2)'/(2*ker_param^2));
   end
case {'Poly', 'Linear'}
   if strcmp(kernel, 'Linear')
      ker_param = 1;
   end
   
   for i = 1:Nf,
      y(:,i) = (train_features'*train_features(:,i) + 1).^ker_param;
   end
case 'Sigmoid'
    if (length(ker_param) ~= 2)
        error('This kernel needs two parameters to operate!')
    end
    
   for i = 1:Nf,
      y(:,i) = tanh(train_features'*train_features(:,i)*ker_param(1)+ker_param(2));
   end
otherwise
   error('Unknown kernel. Can be Gauss, Linear, Poly, or Sigmoid.')
end

%Find the SVM coefficients
switch solver
case 'Quadprog'
   %Quadratic programming
   alpha_star	= quadprog(diag(z)*y'*y*diag(z), -ones(1, Nf), zeros(1, Nf), 1, z, 0, zeros(1, Nf), slack*ones(1, Nf))';
   a_star		= (alpha_star.*z)*y';
   
   %Find the bias
   sv           = find((alpha_star > 0.001*slack) & (alpha_star < slack - 0.001*slack));
   if isempty(sv),
       bias     = 0;
   else
	   B        = z(sv) - a_star(sv);
       bias     = mean(B);
   end
   
case 'Perceptron'
   max_iter		= 1e5;
   iter			= 0;
   rate        = 0.01;
   xi				= ones(1,Nf)/Nf*slack;
   
   if ~isfinite(slack),
       slack = 0;
   end
   
   %Find a start point
   processed_y	= [y; ones(1,Nf)] .* (ones(Nf+1,1)*z);
   a_star		= mean(processed_y')';
   
   while ((sum(sign(a_star'*processed_y+xi)~=1)>0) & (iter < max_iter))
      iter 		= iter + 1;
      if (iter/5000 == floor(iter/5000)),
         disp(['Working on iteration number ' num2str(iter)])
      end
      
      %Find the worse classified sample (That farthest from the border)
      dist			= a_star'*processed_y+xi;
      [m, indice] = min(dist);
      a_star		= a_star + rate*processed_y(:,indice);
      
      %Calculate the new slack vector
      xi(indice)  = xi(indice) + rate;
      xi				= xi / sum(xi) * slack;
   end
   
   if (iter == max_iter),
      disp(['Maximum iteration (' num2str(max_iter) ') reached']);
   else
      disp(['Converged after ' num2str(iter) ' iterations.'])
	end
   
   bias   = 0; 
   a_star = a_star(1:Nf)';
   sv	  = find(abs(a_star) > slack*1e-3);
   
case 'Lagrangian'
    %Lagrangian SVM (See Mangasarian & Musicant, Lagrangian Support Vector Machines)
    tol         = 1e-5;
    max_iter    = 1e5;
    nu          = 1/Nf;
    iter        = 0;

    D           = diag(z);
    alpha       = 1.9/nu;
    
    e           = ones(Nf,1);
    I           = speye(Nf);
    Q           = I/nu + D*y'*D;
    P           = inv(Q);
    u           = P*e;
    oldu        = u + 1;
    
    while ((iter<max_iter) & (sum(sum((oldu-u).^2)) > tol)),
        iter    = iter + 1;
        if (iter/5000 == floor(iter/5000)),
           disp(['Working on iteration number ' num2str(iter)])
        end
        oldu    = u;
        f       = Q*u-1-alpha*u;
        u       = P*(1+(abs(f)+f)/2);
    end
  
    a_star    = y*D*u(1:Nf);
    bias      = -e'*D*u;  
    sv		  = find(abs(a_star) > slack*1e-3);
    
otherwise
   error('Unknown solver. Can be either Quadprog or Perceptron')
end

%Find support verctors
Nsv	    = length(sv);
if isempty(sv),
   error('No support vectors found');
else
   disp(['Found ' num2str(Nsv) ' support vectors'])
end

%Margin
b	= 1/sqrt(sum(a_star.^2));
disp(['The margin is ' num2str(b)])

%Now build the decision region
N           = region(5);
xx	        = linspace (region(1),region(2),N);
yy          = linspace (region(3),region(4),N);
D 			= zeros(N);

for j = 1:N,
   y  = zeros(N,1);
   for i = 1:Nsv,
      data		 = [xx(j)*ones(1,N); yy; ones(1,N)];
      switch kernel,
		case {'Gauss','RBF'},
         y		    = y + a_star(sv(i)) * exp(-sum((data-train_features(:,sv(i))*ones(1,N)).^2)'/(2*ker_param^2));
		case {'Poly', 'Linear'}
         y		    = y + a_star(sv(i)) * (data'*train_features(:,sv(i))+1).^ker_param;
		case 'Sigmoid'
         y		    = y + a_star(sv(i)) * tanh(data'*train_features(:,sv(i))*ker_param(1)+ker_param(2));
      end
   end
   D(:,j) = y;
end

D = D+bias>0;
