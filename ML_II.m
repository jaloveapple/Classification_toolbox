function D = ML_II(train_features, train_targets, Ngaussians, region)

% Classify using the ML-II algorithm. This function accepts as inputs the maximum number
% of Gaussians per class and returns a decision surface based on the most likely number of
% Gaussians in each class
%
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	Ngaussians  - Number of redraws
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%
% Strongly built for only two classes!

[Ndim, M]	= size(train_features);
P_D_given_h	= zeros(2, max(Ngaussians));

%Use holdout for diffrentiating between training data for finding the gaussian parameters
%and the likelihood
holdout			= 0.5;
i					= randperm(M);
train_features = train_features(:,i);
train_targets	= train_targets(i);
EMindices		= 1:floor(M*holdout);
MLindices		= floor(M*holdout)+1:M;
i0					= MLindices(find(train_targets(MLindices) == 0));
i1					= MLindices(find(train_targets(MLindices) == 1));
Ni0				= length(i0);
Ni1				= length(i1);

for i = 1:max(Ngaussians),
   
   %Calculate decision region and error for each possible model. Use only EMindices data
   G				= min([ones(1,length(Ngaussians))*i; Ngaussians]);
   [D, param]	= EM(train_features(:,EMindices), train_targets(:,EMindices), G, region);
   
   %Calculate likelihood of the data given these Gaussians
   %Use only the MLindices data
   if (P_D_given_h(1, G(1)) == 0),	%Do it only if it wasn't already computed
		P_D_given_h(1, G(1)) = computeML(train_features(:,i0), param.m0, param.s0, param.w0);      
   end
   if (P_D_given_h(2, G(2)) == 0),	%Do it only if it wasn't already computed
		P_D_given_h(2, G(2)) = computeML(train_features(:,i1), param.m1, param.s1, param.w1);      
   end
   
end

likelihood = P_D_given_h(1,[1:Ngaussians(1)])' * P_D_given_h(2,[1:Ngaussians(2)]);

%Choose the ML model as the one with the lowest error
[i1, i2] = find(likelihood == max(max(likelihood)));
i1       = i1(1); i2 = i2(1); %To give preference for simpler models...

D        = EM(train_features, train_targets, [i1, i2], region);

disp(['FINAL SELECTION: Using ' num2str(i1) ' Gaussians for class 1 and ' num2str(i2) ' Gaussians for class 2'])


function P = computeML(features, mu, sigma, w)

M	= size(features,2);
Ng	= size(mu,1);
p	= zeros(Ng, M);

warning off

if Ng == 1,
	for j = 1:M,
   	x		= features(:,j);
      p(j)	= w(1)/(2*pi*sqrt(det(sigma)))*exp(-0.5*(x-mu')'*inv(sigma)*(x-mu'));
   end
   P			= prod(p);
else
   for j = 1:M,
   	x		= features(:,j);
      for k = 1:length(w),
         p(k, j) = w(k)/(2*pi*sqrt(det(squeeze(sigma(k,:,:)))))*...
            exp(-0.5*(x-mu(k,:)')'*inv(squeeze(sigma(k,:,:)))*(x-mu(k,:)'));
      end
   end
   P			= prod(sum(p));   
end

