function [D, w] = Stumps(train_features, train_targets, params, region)

% Classify using the least-squares algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	weights	- Unused (Except if weighted stumps is needed)
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%	w			- Decision surface parameters

train_one  = find(train_targets == 1);
train_zero = find(train_targets == 0);

if (length(params)-1 == length(train_targets)), 
    p = params(1:end-1);
else
    p = ones(size(train_targets));   
end

dim        = size(train_features,1);
w          = zeros(1,dim);
err        = zeros(1,dim);
direction  = zeros(1,dim);

for i = 1:dim,
    %For each dimension, find the point where a stump gives the minimal error
    
    %First, sort the working dimension 
    [data(i,:), indices] = sort(train_features(i,:));
    temp_targets    = train_targets(indices);
    temp_p		 	  = p(indices);
    
    decision        = cumsum(temp_p .* temp_targets)/length(train_one) - cumsum(temp_p .* (~temp_targets))/length(train_zero);
    [err(i),W]      = max(abs(decision));
    w(i)            = data(i,W);
    direction(i)    = sign(decision(W));
end

[m, min_dim] = max(err);
indices      = find(~ismember(1:dim,min_dim));
w(indices)   = 0;

N    		 = region(5);

if (dim == 2),
    %Find decision region (For 2-D data)
    x		= linspace (region(1),region(2),N);
    y		= linspace (region(3),region(4),N);
    D       = zeros(N);
    
    if (w(1)~=0),
        ix = find(data(1,:)==w(1)); ix = ix(1);
        if ix == length(data(1,:)),
            xt = region(2);
        else
            xt = (data(1,ix+1) + data(1,ix)) / 2; 
        end
        [m, indice] = min(abs(x - xt));
        if (direction(1) < 0),
            D(:,indice+1:N) = 1;
        else
            D(:,1:indice) = 1;
        end
    else
        iy = find(data(2,:)==w(2)); iy = iy(1);
        if iy == length(data(2,:)),
            yt = region(4);
        else
            yt = (data(2,iy+1) + data(2,iy)) / 2; 
        end
        [m, indice] = min(abs(y - yt));
        if (direction(2) < 0),
            D(indice+1:N,:) = 1;
        else
            D(1:indice,:) = 1;
        end
    end     
else
    D = zeros(N);
    disp('No decision region calculated because the data has more than two dimensions')
end