function [features, targets, label] = BIMSEC(train_features, train_targets, params, region, plot_on)

%Reduce the number of data points using the basic iterative MSE clustering algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	params			- Algorithm parameters: [Number of output data points, Number of attempts]
%	region			- Decision region vector: [-x x -y y number_of_points]
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets
%	label				- The labels given for each of the original features

if (nargin < 5),
    plot_on = 0;
end
[Nmu, Ntries] = process_params(params);

[D,L]	= size(train_features);
dist	= zeros(Nmu,L);
label   = zeros(1,L);

%Initialize the mu's
mu			= randn(D,Nmu);
mu			= sqrtm(cov(train_features',1))*mu + mean(train_features')'*ones(1,Nmu);
ro          = zeros(1,Nmu);
n           = zeros(1,Nmu);
Ji          = zeros(1,Nmu);
oldJ        = 0;
J           = 1;

if (Nmu == 1),
   mu		= mean(train_features')';
   label	= ones(1,L);
else  
    %Assign each example to one of the mu's
    %Compute distances
    dist    = zeros(Nmu, L);
    for i = 1:Nmu,
        dist(i,:) = sqrt(sum((mu(:,i)*ones(1,L) - train_features).^2));
    end
    [m, label]  = min(dist);
    n           = hist(label, Nmu);

    while (Ntries > 0),
        
        %Select a sample x_hat  
        r     = randperm(L);
        x_hat = train_features(:,r(1));
        
        %i <- argmin||mi - x_hat||
        dist  = sqrt(sum((mu - x_hat * ones(1,Nmu)).^2));
        i     = find(dist == min(dist));
        
        %Compute ro if n(i) ~= 1
        if (n(i) ~=1),
            for j = 1:Nmu,
                if (i ~= j),
                    ro(j) = n(j)/(n(j)+1)*dist(j)^2;
                else
                    ro(j) = n(j)/(n(j)-1)*dist(j)^2;
                end
            end
            
            %Transfer x_hat if needed
            [m, k] = find(min(ro) == ro);
            if (k ~= i),
                label(r(1)) = k;
                n(i)        = n(i) - 1;
                n(k)        = n(k) + 1;
                
                %Recompute Je, and the mu's
                for j = 1:Nmu,
                    indexes = find(label == j);
                    mu(:,j) = mean(train_features(:,indexes)')';
                    Ji(j)   = sum(sum((mu(:,j)*ones(1,length(indexes)) - train_features(:,indexes)).^2));
                end
                
                oldJ    = J;
                J       = sum(Ji);
            end
                
        end
                 
        %disp(['Distance to convergence is ' num2str(abs(J-oldJ))])
        if (plot_on == 1),
            plot_process(mu)
        end
 
        if (J == oldJ),
            Ntries = Ntries - 1;
        end

    end
end

%Make the decision region
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