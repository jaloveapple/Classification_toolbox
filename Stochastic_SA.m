function [features, targets] = Stochastic_SA(train_features, train_targets, params, region, plot_on)

%Reduce the number of data points using the stochastic simulated annealing algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	params	    	- [Number of output data points, cooling rate (Between 0 and 1)]
%	region			- Decision region vector: [-x x -y y number_of_points]
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets

if (nargin < 5),
    plot_on = 0;
end

%Parameters:
[Nmu, epsi]	 = process_params(params);
T		= max(eig(cov(train_features',1)'))/2;    %Initial temperature
Tmin    = 0.01;                                 %Stopping temperature

[d,L]	= size(train_features);
label   = zeros(1,L);
dist	= zeros(Nmu,L);
iter    = 0;
max_change = 1e-3;

%Initialize the mu's
mu			= mean(train_features')';

%Init the inclusion matrix
inclusion_mat   = rand(Nmu, L);
[m, i]          = max(inclusion_mat);
inclusion_mat   = zeros(Nmu,L);
for j = 1:L,
    inclusion_mat(i(j),j) = 1;
end

if (Nmu >= 1),
    while (T > Tmin),  
        iter    = iter + 1;
        index   = randperm(L);
        T = epsi * T;
        
        for i = 1:L,
            %Select a node (example) randomally. Poll all nodes once

            %Calculate the energy in this configuration: Ea <- 1/2*sum(w_ij*s_i*s_j)
            Ea = energy(train_features, inclusion_mat);
    
            %Change the configuration and see what the energy is
            config = inclusion_mat(:,index(i));
            change = rand(Nmu,1);
            [m, j] = max(~config.*change);
            new_inclusion_mat               = inclusion_mat;
            new_inclusion_mat(:,index(i))   = 0;
            new_inclusion_mat(j,index(i))   = 1;
            Eb     = energy(train_features, new_inclusion_mat);
            
            if (Eb < Ea),
                inclusion_mat = new_inclusion_mat;
            else
                if (exp(-(Eb-Ea)/T) > rand(1)),
                    inclusion_mat = new_inclusion_mat;
                end
            end
        end
        
        %Recalculate the mu's
        mu  = zeros(d, Nmu);
        for i = 1:Nmu,
            indices = find(inclusion_mat(i,:) == 1);
            mu(:,i) = mean(train_features(:,indices)')';
        end
        if (plot_on == 1),
            plot_process(mu)
        end
    
    end    
end


%Make the decision region
dist	= zeros(Nmu,L);
for i = 1:Nmu,
   dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
end
[m,label] = min(dist);

targets = zeros(1,Nmu);
if (Nmu > 1),
	for i = 1:Nmu,
   	    if (length(train_targets(:,find(label == i))) > 0),
          	targets(i) = (sum(train_targets(:,find(label == i)))/length(train_targets(:,find(label == i))) > .5);
   	    end
	end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end

features = mu;



function E = energy(features, inclusion_matrix)

%Calculate the energy value given the features and the inclusion matrix
%The energy function tries to minimize the in-class variance

[N,M]   = size(inclusion_matrix);
e       = zeros(1,N+1);

for i = 1:N,
    indices = find(inclusion_matrix(i,:) == 1);
    mu      = mean(features(:,indices)')';
    e(i)    = sum(sum((features(:,indices) - mu*ones(1,length(indices))).^2));
end

E = sum(e);