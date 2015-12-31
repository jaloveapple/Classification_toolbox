function [D, P, theta, phi] = LocBoost(features, targets, params, region)

% Classify using the local boosting algorithm
% Inputs:
% 	features	- Train features
%	targets		- Train targets
%   params      - A vector containing the algorithm paramters:
%                 [Number of boosting iterations, number of EM iterations, Number of optimization iterations, Optimize using a test set {0/1}, Weak learner, Weak learner parameters]
%                 IMPORTANT: The weak learner must return a hyperplane parameter vector, as in LS
%	region		- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   P           - The probability function (NOT the probability for the train_targets!)
%	theta		- Sub-classifier parameters
%	phi		    - Sub-classifier weights

test_percentage = 0.1;                  %Percentage of points to be used as a test set
[Dims, Nf]      = size(features);
train_indices   = 1:Nf;
test_indices    = [];
Nt              = 0;
targets		    = (targets > .5)*2-1;	%Set targets to {-1,1}
opt_error       = [1 1];
[Niterations, Nem, Noptim, LowerBound, optimize, Wtype, Wparams] = process_params(params);
Niterations		 = Niterations + 1;

UpperBound = 1/LowerBound;

if ((Niterations < 1) | (Nem < 1) | (Noptim < 1)),
   error('Iteration paramters must be positive!');
end

options		= optimset('Display', 'off', 'MaxIter', Noptim);
if (optimize == 1),
   %Perform optimization
   [test_indices, train_indices] = make_a_draw(floor((1-test_percentage)*Nf), Nf);
   Nf           = length(train_indices);
   Nt           = length(test_indices);
end
  
warning off;

%For decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:), ones(N^2,1)]';

%Find first iteration parameters
theta     	= zeros(Niterations, Dims+1);
phi			= zeros(Niterations, Dims*2);
h			= ones(1,Nf);

phi(:,Dims+1:2*Dims) = ones(Niterations,Dims);

%Initial value is the largest connected component again all the others
[Iphi, dist, indices]	= compute_initial_value(features(:,train_indices), targets(train_indices), []);
[D, t_theta]            = feval(Wtype, features(:,train_indices), targets(train_indices)>0, Wparams, region);
theta(1, 1:size(t_theta, 2)) = t_theta;

class_ker   = LocBoostFunctions(theta(1,:), 'class_kernel', [features(:,train_indices); ones(1,Nf)], targets(train_indices)); 
P           = class_ker;
Pdecision   = LocBoostFunctions(theta(1,1:3), 'class_kernel', flatxy, ones(1,N^2)); 

if (optimize == 1),
    %Optimization needed
    Poptimization   = LocBoostFunctions(theta(1,:), 'class_kernel', [features(:,test_indices); ones(1,Nt)], targets(test_indices)); 
else
    Poptimization   = [];
end

%Find the local classifiers
for t = 2:Niterations,
    
    %Do inital guesses
    [Iphi, dist, indices]	= compute_initial_value(features(:,train_indices), (P<0.5), dist);
    
    if ~isfinite(sum(Iphi)),
        in = find(~isfinite(Iphi));
        Iphi(in) = 0;
        warning('Infinite initial guess')
    end   

    %If there is something more to do ...
    if ~isempty(indices),
        if (length(indices) > 2),
            phi(t,:)    = Iphi;
        else
            phi         = phi(1:t-1,:);
            theta       = theta(1:t-1,:);
            disp(['No more indices to work on. Largest connected component is ' num2str(length(indices)) ' long.'])
            break;
        end
    end    
    
    [D, t_theta] = feval(Wtype, features(:,train_indices).*(ones(Dims,1)*LocBoostFunctions(phi(t,:), 'gamma_kernel', features(:,train_indices))),targets(train_indices)>0, Wparams, region);
    theta(t, 1:size(t_theta, 2)) = t_theta;

    opt_error(2) = 1;
    for i = 1:Nem,
        %Compute h(t-1)
	    gamma_ker 	    = LocBoostFunctions(phi(t,:), 'gamma_kernel', features(:,train_indices));  %Gamma(x, gamma(C))
       	class_ker      = LocBoostFunctions(theta(t,:), 'class_kernel', [features(:,train_indices); ones(1,Nf)], targets(train_indices)); 
	    h_tminus1      = gamma_ker .* class_ker ./ ((1-gamma_ker).*P + gamma_ker.*class_ker);
    
    	%Optimize theta(t,:) using first part of the Q function
	    temp_theta     = fminsearch('LocBoostFunctions', theta(t,:), options, 'Q1', [features(:,train_indices); ones(1,Nf)], targets(train_indices), h_tminus1);
        %[d, temp_theta(1,1:size(theta,2))] = feval('LS', features(:,train_indices), targets(train_indices), h_tminus1, region);
        
    	%Optimize gamma(t,:) using second part of the Q function
        temp_phi       = fminsearch('LocBoostFunctions', phi(t,:), options, 'Q2',  features(:,train_indices), targets(train_indices), h_tminus1);

        if (optimize == 1),
            %Optimization needed
            opt_gamma = LocBoostFunctions(temp_phi, 'gamma_kernel', features(:,test_indices));
            opt_class = LocBoostFunctions(temp_theta, 'class_kernel', [features(:,test_indices); ones(1,Nt)], targets(test_indices));
            temp_Poptimization   = (1-opt_gamma).*Poptimization + opt_gamma.*opt_class;
            new_error = sum((temp_Poptimization>0.5)~=(targets(test_indices)>0))/Nt;
            if (new_error < opt_error(2)),
                %Error got lower
                opt_error(2)    = new_error;
                theta(t,:)      = temp_theta;
                phi(t,:)        = temp_phi;
            else
                if (new_error*0.9 > opt_error)
                    %The error got larger
                    break
                end
            end
        end
        
        theta(t,:) = temp_theta;
        phi(t,:)   = temp_phi;
    end
    disp(num2str(max(phi(t,Dims+1:2*Dims))))
    
    phi(t, Dims+1:2*Dims)       = min(UpperBound, phi(t, Dims+1:2*Dims));
    
    %Compute new P function 
    gamma_ker	   = LocBoostFunctions(phi(t,:), 'gamma_kernel', features(:,train_indices));  
    class_ker     = LocBoostFunctions(theta(t,:), 'class_kernel', [features(:,train_indices); ones(1,Nf)], targets(train_indices)); 
    P             = (1-gamma_ker).*P + gamma_ker.*class_ker;
    
    if (optimize == 1),
        %Optimization needed
        opt_gamma = LocBoostFunctions(phi(t,:), 'gamma_kernel', features(:,test_indices));
        opt_class = LocBoostFunctions(theta(t,:), 'class_kernel', [features(:,test_indices); ones(1,Nt)], targets(test_indices));
        Poptimization   = (1-opt_gamma).*Poptimization + opt_gamma.*opt_class;
        new_error = sum((Poptimization>0.5)~=(targets(test_indices)>0))/Nt;
        if (new_error < opt_error(1)),
            opt_error(1) = new_error;
        else
            if (new_error*0.7 > opt_error(1))
                %The error got larger
                phi         = phi(1:t-1,:);
                theta       = theta(1:t-1,:);
                disp('Validation error got much larger')
                break
            end
        end
    end
    
    Dgamma		   = LocBoostFunctions(phi(t,[1:2,Dims+[1:2]]), 'gamma_kernel', flatxy(1:2,:));  %Gamma(x, gamma(C))
    Dclass			= LocBoostFunctions(theta(t,1:3), 'class_kernel', flatxy, ones(1,N^2));
    Pdecision      = (1-Dgamma).*Pdecision + Dgamma.*Dclass;
    
    %disp(['Finished iteration number ' num2str(t-1)])

    if (sum(P<.5) == 0),
       %Nothing more to do
       phi         = phi(1:t,:);
       theta       = theta(1:t,:);
       disp('P=0.5 for all indices')
       break
    end
    
end

if (Dims == 2),
   %Find decision region (********** Made for only 2 classes ************)
	Dnot    = reshape((Pdecision>.499)&(Pdecision<.501),N,N);
	Dnn     = Nearest_Neighbor(features, targets>0, 3, region);
	D       = ~Dnot.*reshape((Pdecision>=.5), N, N) + Dnot .* Dnn;
else
   %disp('Decision surface can only be used for two dimensional data.')
   %disp(['This data has ' num2str(Dims) ' dimensions, so please disregard the decision surface'])
end


newP    = zeros(1,Nf+Nt);
newP(train_indices) = P;
newP(test_indices) = Poptimization;
P       = newP;
%end LocBoost
%*********************************************************************

function [phi, dist, indices] = compute_initial_value(features, targets, dist)

%Returns the initial guess by connected components

[Dim,n] = size(features);

% Compute all distances, if it has not been done before
if (isempty(dist)),
   for i = 1:n,
      dist(i,:) = sum((features(:,i)*ones(1,n) - features).^2);
   end
end

ind_plus	= find(targets == 1);
size_plus   = length(ind_plus);

G = zeros(n);
for i=1:size_plus   
   [o,I] = sort(dist(ind_plus(i),:));
   for j=1:n
      if (targets(I(j)) == 1),
         G(ind_plus(i),I(j)) = 1;
         G(I(j),ind_plus(i)) = 1;
      else
         break
      end
   end
end
G = G - (tril(G).*triu(G)); %Remove main diagonal

if ~all(diag(G)) 
    [p,p,r,r] = dmperm(G|speye(size(G)));
else
    [p,p,r,r] = dmperm(G);  
end;
 
% Now the i-th component of G(p,p) is r(i):r(i+1)-1.
sizes   = diff(r);        % Sizes of components, in vertices.
k       = length(sizes);      % Number of components.
 
% Now compute an array "blocks" that maps vertices of G to components;
% First, it will map vertices of G(p,p) to components...
component           = zeros(1,n);
component(r(1:k))   = ones(1,k);
component           = cumsum(component);
 
% Second, permute it so it maps vertices of A to components.
component(p) = component;

[n1, n2]	 = hist(component, unique(component));
[m, N]   	 = max(n1);
indices		 = find(component == n2(N));
means			 = mean(features(:,indices)');
stds     	 = std(features(:,indices)');
phi 			 = [means, 1./stds.^2];

%End
