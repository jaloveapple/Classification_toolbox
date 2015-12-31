function D = Multivariate_Splines(train_features, train_targets, params, region)

% Classify using multivariate adaptive regression splines
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	params      - [Spline degree, Number of knots per spline]
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[Ni, M]		    = size(train_features);
train_targets   = train_targets*2-1;
means           = mean(train_targets);
train_targets   = train_targets - means;
options         = optimset('Display','off');

%Get parameters
[q, Nk]			 = process_params(params);

order           = zeros(1,Ni); %The new order of the dimensions
residual        = train_targets;
knots           = zeros(Nk, Ni);

%Define what a spline is
Pspline         = inline('sum(((x-t*ones(1,L)).^q).*(((x-t*ones(1,L)).^q)>0))', 't', 'x', 'q', 'L');
minPspline      = inline('sum((targets - sum(((x-t*ones(1,L)).^q).*(((x-t*ones(1,L)).^q)>0))).^2)/L', 't', 'x', 'q', 'L', 'targets');

for i = 1:Ni,
    %Find which remaining dimension is fit by a spline:
    
    %Findd the remaining dimensions
    remaining = find(~ismember(1:Ni, order));
    
    %Fit a spline to each of the dimensions
    temp_knots  = zeros(Nk, Ni-i+1);
    errors      = zeros(1, Ni-i+1);
    for j = 1:Ni-i+1,
        temp_knots(:,j) = fminunc(minPspline, randn(Nk, 1), options, ones(Nk,1)*train_features(remaining(j),:), q, M, residual);
        errors(j)       = feval(minPspline, temp_knots(:,j), ones(Nk,1)*train_features(remaining(j),:), q, M, residual);
    end
    
    %Find the best dimension to regress on
    [best, best_dim] = min(errors);
    order(i)         = remaining(best_dim);
    knots(:,i)       = temp_knots(:,best_dim);
    
    %Compute residual 
    predict          = feval(Pspline, temp_knots(:,j), ones(Nk,1)*train_features(remaining(j),:), q, M);
    residual         = residual ./ predict;
end

%Compute weights via pseudo-inverse:
%Compute the prediction for each dimension
prediction = zeros(Ni, M);
for i = 1:Ni,
    prediction(i,:) = feval(Pspline, knots(:,i), ones(Nk,1)*train_features(order(i),:), q, M);
    if i > 1,
        prediction(i,:) = prod(prediction(1:i,:));
    end
end

%Compute the weights
W   = pinv(prediction*prediction')*prediction*train_targets';

%Compute the decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:)]';

prediction  = zeros(Ni, N^2);
for i = 1:Ni,
    prediction(i,:) = feval(Pspline, knots(:,i), ones(Nk,1)*flatxy(order(i),:), q, N^2);
    if i > 1,
        prediction(i,:) = prod(prediction(1:i,:));
    end
end

d   = W'*prediction + means;
D   = reshape(d,N,N)>0;
