function D = ada_boost(train_features, train_targets, params, region);

% Classify using the AdaBoost algorithm
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Params	- [NumberOfIterations, Weak Learner Type, Learner's parameters]
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%
% NOTE: This algorithm is very tuned to the 2D nature of the toolbox!

[k_max, weak_learner, alg_param] = process_params(params);

[Ni,M]			= size(train_features);
D			 		= zeros(region(5));
W			 		= ones(1,M)/M;
IterDisp			= 10;

%Find where the training features fall on the decision grid
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:)]';
train_loc	= zeros(1,M);
for i = 1:M,
   dist = sqrt(sum((flatxy - train_features(:,i)*ones(1,N^2)).^2));
   [m, train_loc(i)] = min(dist);
end

%Do the AdaBoosting
for k = 1:k_max,
   %Train weak learner Ck using the data sampled according to W:
   %...so sample the data according to W
   randnum = rand(1,M);
   cW	   = cumsum(W);
   indices = zeros(1,M);
   for i = 1:M,
      %Find which bin the random number falls into
      loc = max(find(randnum(i) > cW))+1;
      if isempty(loc)
         indices(i) = 1;
      else
         indices(i) = loc;
      end
   end
   
   %...and now train the classifier
   Ck 	= feval(weak_learner, train_features(:, indices), train_targets(indices), alg_param, region);
   Ckl 	= Ck(:);
   
   %Ek <- Training error of Ck 
   Ek = sum(W.*(Ckl(train_loc)' ~= train_targets));
   
   if (Ek == 0),
      break
   end
   
   %alpha_k <- 1/2*ln(1-Ek)/Ek)
   alpha_k = 0.5*log((1-Ek)/Ek);
   
   %W_k+1 = W_k/Z*exp(+/-alpha)
   W  = W.*exp(alpha_k*(xor(Ckl(train_loc)',train_targets)*2-1));
   W  = W./sum(W);
   
   %Update the decision region
   D  = D + alpha_k*(2*Ck-1);
   
   if (k/IterDisp == floor(k/IterDisp)),
      disp(['Completed ' num2str(k) ' boosting iterations'])
   end
   
end

D = D>0;