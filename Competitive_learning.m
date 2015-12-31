function [features, targets, label, W] = Competitive_learning(train_features, train_targets, params, region, plot_on)

% Perform preprocessing using a competitive learning network
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	params	    - [Number of partitions, learning rate]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%   plot_on     - Plot while performing processing?
%
% Outputs
%	features		- New features
%	targets			- New targets
%	label			- The labels given for each of the original features
%   W               - Weights matrice

max_iter       = 1000;
[c, r]		   = size(train_features);
[N, eta]			= process_params(params);
decay          = 0.99;

%Preprocessing:
% x_i <- {x_i, 1}
x              = [train_features ; ones(1,r)];
%x_i <- x_i./||x_i||
x              = x ./ (ones(c+1,1) * sqrt(sum(x.^2)));

%Initialize the W's
i              = randperm(r);
W              = x(:,i(1:N));

for i = 1:max_iter,
    %Randomally order the patterns
    order = randperm(r);
    change= 0;
    
    for k = 1:r,
        J = W'*x(:,order(k));
        j = find(J == max(J));
        
        old_W   = W(:,j);
        
        %W_j <- W_j + eta*x
        W(:,j)  = W(:,j) + eta*x(:,order(k));
        
        %W_j <- W_j/||W_j||
        W(:,j)  = W(:,j) / sqrt(sum(W(:,j).^2));
        
        change = change + sum(abs(W(:,j) - old_W));
        
        if (plot_on == 1),
            %Assign each of the features to a center
            dist        = W'*x;
            [m, label]  = max(dist);
            centers     = zeros(c,N);
            for i = 1:N,
                in = find(label == i);
                if ~isempty(in)
                    centers(:,i) = mean(x(1:2,find(label==i))')';
                else
                    centers(:,i) = nan;
                end
            end
            plot_process(centers)
        end

    end

    eta = eta * decay;
    
    if (change/r < 1e-4),
        break
    end
    
end

if (i == max_iter),
   disp(['Maximum iteration (' num2str(max_iter) ') reached']);
else
    disp(['Finished after ' num2str(i) ' iterations.'])
end

%Assign each of the features to a center
dist        = W'*x;
[m, label]  = max(dist);
features     = zeros(c,N);
for i = 1:N,
    in = find(label == i);
    if ~isempty(in)
        features(:,i) = mean(x(1:2,find(label==i))')';
    else
        features(:,i) = nan;
    end
end

targets = zeros(1,N);
if (N > 1),
	for i = 1:N,
   	if (length(train_targets(find(label == i))) > 0),
      	targets(i) = (sum(train_targets(find(label == i)))/length(train_targets(find(label == i))) > .5);
   	end
	end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end
