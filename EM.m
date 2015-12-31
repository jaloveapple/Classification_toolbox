function [D, param_struct] = EM(train_features, train_targets, Ngaussians, region)

% Classify using the expectation-maximization algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%  Ngaussians   - Number for Gaussians for each class (vector)
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   param_struct- A parameter structure containing the parameters of the Gaussians found


[Nclasses, classes]  = find_classes(train_targets); %Number of classes in targets
Nalpha				 = Ngaussians;						 %Number of Gaussians in each class

max_iter   			= 100;
max_try             = 5;
Pw					= zeros(2,max(Ngaussians));
sigma				= zeros(2,max(Ngaussians),size(train_features,1),size(train_features,1));
mu					= zeros(2,max(Ngaussians),size(train_features,1));

%The initial guess is based on k-means preprocessing. If it does not converge after
%max_iter iterations, a random guess is used.
disp('Using k-means for initial guess')
for i = 1:Nclasses,
    in  			= find(train_targets==classes(i));
    [initial_mu, targets, labels]	= k_means(train_features(:,in),train_targets(:,in),Ngaussians(i),region);
    for j = 1:Ngaussians(i),
        gauss_labels    = find(labels==j);
        Pw(i,j)         = length(gauss_labels) / length(labels);
        sigma(i,j,:,:)  = diag(std(train_features(:,in(gauss_labels))'));
    end
    mu(i,1:Ngaussians(i),:) = initial_mu';
end

%Do the EM: Estimate mean and covariance for each class 
for c = 1:Nclasses,
    train	   = find(train_targets == classes(c));
    
    if (Ngaussians(c) == 1),
        %If there is only one Gaussian, there is no need to do a whole EM procedure
        sigma(c,1,:,:)  = sqrtm(cov(train_features(:,train)',1));
        mu(c,1,:)       = mean(train_features(:,train)');
    else
        
        sigma_i         = squeeze(sigma(c,:,:,:));
        old_sigma       = zeros(size(sigma_i)); 		%Used for the stopping criterion
        iter			= 0;									%Iteration counter
        n			 	= length(train);					%Number of training points
        qi			    = zeros(Nalpha(c),n);	   	%This will hold qi's
        P				= zeros(1,Nalpha(c));
        Ntry            = 0;
        
        while ((sum(sum(sum(abs(sigma_i-old_sigma)))) > 1e-10) & (Ntry < max_try))
            old_sigma = sigma_i;
            
            %E step: Compute Q(theta; theta_i)
            for t = 1:n,
                data  = train_features(:,train(t));
                for k = 1:Nalpha(c),
                    P(k) = Pw(c,k) * p_single(data, squeeze(mu(c,k,:)), squeeze(sigma_i(k,:,:)));
                end          
                
                for i = 1:Nalpha(c),
                    qi(i,t) = P(i) / sum(P);
                end
            end
            
            %M step: theta_i+1 <- argmax(Q(theta; theta_i))
            %In the implementation given here, the goal is to find the distribution of the Gaussians using
            %maximum likelihod estimation, as shown in section 10.4.2 of DHS
            
            %Calculating mu's
            for i = 1:Nalpha(c),
                mu(c,i,:) = sum((train_features(:,train).*(ones(2,1)*qi(i,:)))')/sum(qi(i,:)');
            end
            
            %Calculating sigma's
            %A bit different from the handouts, but much more efficient
            for i = 1:Nalpha(c),
                data_vec = train_features(:,train);
                data_vec = data_vec - squeeze(mu(c,i,:)) * ones(1,n);
                data_vec = data_vec .* (ones(2,1) * sqrt(qi(i,:)));
                sigma_i(i,:,:) = sqrt(abs(cov(data_vec',1)*n/sum(qi(i,:)')));
            end
            
            %Calculating alpha's
            Pw(c,1:Ngaussians(c)) = 1/n*sum(qi');
            
            iter = iter + 1;
            disp(['Iteration: ' num2str(iter)])
            
            if (iter > max_iter),
                theta = randn(size(sigma_i));
                iter  = 0;
                Ntry  = Ntry + 1;
                
                if (Ntry > max_try)
                    disp(['Could not converge after ' num2str(Ntry-2) ' redraws. Quitting']);
                else
                    disp('Redrawing weights.')
                end
            end
            
        end
        
        sigma(c,:,:,:) = sigma_i;
    end
end

%Find decision region
p0				= length(find(train_targets == 0))/length(train_targets);

%If there is only one gaussian in a class, squeeze will wreck it's format, so fixing is needed
m0  = squeeze(mu(1,1:Ngaussians(1),:));
m1  = squeeze(mu(2,1:Ngaussians(2),:));
if (size(m0,2) == 1),
    m0 = m0';
end
if (size(m1,2) == 1),
    m1 = m1';
end

param_struct.m0 = m0;
param_struct.m1 = m1;
param_struct.s0 = squeeze(sigma(1,1:Ngaussians(1),:,:));
param_struct.s1 = squeeze(sigma(2,1:Ngaussians(2),:,:));
param_struct.w0 = Pw(1,1:Ngaussians(1),:);
param_struct.w1 = Pw(2,1:Ngaussians(2),:);
param_struct.p0 = p0;

D	= decision_region(param_struct, region);

%END EM

function p = p_single(x, mu, sigma)

%Return the probability on a Gaussian probability function. Used by EM

p = 1/(2*pi*sqrt(abs(det(sigma))))*exp(-0.5*(x-mu)'*inv(sigma)*(x-mu));

