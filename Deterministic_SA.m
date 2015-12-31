function [features, targets] = Deterministic_SA(train_features, train_targets, params, region, plot_on)

%Reduce the number of data points using the deterministic simulated annealing algorithm
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
[Nmu, epsi] = process_params(params);

T		= max(eig(cov(train_features',1)'))/2;    %Initial temperature
Tmin    = T/500;                                 %Stopping temperature

[d,L]	= size(train_features);
label   = zeros(1,L);
dist	= zeros(Nmu,L);
iter    = 0;
max_change = 1e-3;

%Init the inclusion matrix
inclusion_mat   = rand(Nmu, L);
inclusion_mat   = inclusion_mat ./ (ones(Nmu,1)*sum(inclusion_mat));

if (Nmu == 1),
    %Initialize the mu's
    mu			= mean(train_features')';
else
    %Initialize the P
    P   = rand(Nmu,L);
    P   = P ./ (ones(Nmu,1)*sum(P));
    
    while (T > Tmin),  
        iter    = iter + 1;
        T = epsi * T;
        
        for i = 1:L,
            %For each node (example):
            %Recompute the mu's
            for i = 1:Nmu,
               mu(:,i) = sum(((ones(d,1)*P(i,:)).*train_features)')'./(sum(P(i,:))); 
            end
            
            %Find the distances from mu's to features 
            for i = 1:Nmu,
               dist(i,:) = sum((train_features - mu(:,i)*ones(1,L)).^2);
            end
            dist = exp(-dist/T);
            %In this implementation, s_i is equal to dist!
               
            %Compute Gibbs distribution
            P = dist ./ (ones(Nmu,1) * sum(dist));
            if (~isfinite(sum(sum(P))))
                disp('P is infinite. Stopping.')
                break
            end
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


