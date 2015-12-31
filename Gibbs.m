function D = Gibbs(train_features, train_targets, Ndiv, region)

% Classify using the Gibbs algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	Ndiv    - Resolution of the distribution of Theta given the data
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[Ni, M] 		= size(train_features);
Uclasses		= unique(train_targets);
Nc		 		= length(Uclasses);


mu      = zeros(Nc, Ni, Ndiv);
sigma   = zeros(Nc, Ni, Ni, Ndiv);      %We will assume diagonal matrices for simplicity
for i = 1:Nc,
    in              = find(train_targets == Uclasses(i));
    
    %Find the range of the data
    min_range   = min(train_features(:,in)')';
    max_range   = max(train_features(:,in)')';
    
    %Make a "guess" on theta based on this range of data
    for j = 1:Ni,
        mu(i, j, :)     = linspace(min_range(j), max_range(j),Ndiv);
        sigma(i,j,j,:)  = linspace(min_range(j), max_range(j),Ndiv);
    end
end

%Now test the probability of the test set given these parameters (P(D|theta))
P_D_given_Theta = zeros(Nc, Ndiv, Ndiv);
X               = train_features;
for i = 1:Nc,
    in              = find(train_targets == Uclasses(i));
    x               = X(:,in);
    pD_given_theta  = zeros(1, length(in));
    m               = length(in);
    
    for j = 1:Ndiv,
        for k = 1:Ndiv,
            temp_mu         = squeeze(mu(i, :, j))';
            temp_si         = squeeze(sigma(i, :, :, k));
            pD_given_theta  = 1/((2*pi*det(temp_si))^(Ni/2)*exp(-0.5*diag((x-temp_mu*ones(1,m))'*inv(temp_si)*(x-temp_mu*ones(1,m)))));
            P_D_given_Theta(i,j,k)  = sum(pD_given_theta);
            %P_D_given_Theta(i,j,k)  = prod(pD_given_theta(find(pD_given_theta~=0)));
        end
    end
end

%Find P(Theta|D), and draw one using it
P_Theta_given_D = zeros(size(P_D_given_Theta));    
Theta_mu        = zeros(Nc, Ni, 1);
Theta_sigma     = zeros(Nc, Ni, Ni);
for i = 1:Nc,
    P_Theta_given_D(i,:,:) = squeeze(P_D_given_Theta(i,:,:))/sum(sum(squeeze(P_D_given_Theta(i,:,:))));
    
    %Make a draw
    n                      = rand(1);
    seq                    = cumsum(reshape(squeeze(P_Theta_given_D(i,:,:)),Ndiv^2,1));
    loc                    = max(find(n > seq))+1;
    if isempty(loc)
       indices(i) = 1;
    else
       indices(i) = loc;
    end
    seq                    = zeros(size(seq)); seq(loc)=1;
    [j,k]                  = find(reshape(seq, Ndiv, Ndiv));
    Theta_mu(i, :, 1)      = squeeze(mu(i, :, j))';
    Theta_sigma(i, :, :)   = squeeze(sigma(i, :, :, k));
end

%Find decision region (For two dimensional problems possible)
param_struct.m0		= squeeze(Theta_mu(1, :, :));
param_struct.s0		= squeeze(Theta_sigma(1, :, :));
param_struct.m1		= squeeze(Theta_mu(2, :, :));
param_struct.s1		= squeeze(Theta_sigma(2, :, :));
param_struct.p0		= sum(~train_targets)/length(train_targets);
param_struct.w0		= 1;
param_struct.w1		= 1;
D		= decision_region(param_struct, region);
