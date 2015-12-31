function D = Interactive_Learning(train_features, train_targets, params, region);

% Classify using nearest neighbors and interactive learning
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params  - [Number of query points, Weight (Relative weight of the new point in relation to the old data points)] 
%
% Outputs
%	D			- Decision sufrace
%
% In this implementation, we train a nearest neighbor classifier, and ask for interactive assistance in problematic areas

[Npoints , Weight] = process_params(params);

Weight  = round(Weight*size(train_features,2));
NN      = 3;

hm      = findobj('Tag', 'Messages'); 
N       = region(5);
x       = linspace(region(1),region(2),N);
y       = linspace(region(3),region(4),N);
D       = NearestNeighbor(train_features, train_targets, NN, region);

for i = 1:Npoints,
    %Find the most ambiguous point in the decision region
    ambig   = abs(D-0.5);
    [iy,ix] = find(ambig == min(min(ambig)));
    ix      = ix(1); iy = iy(1);
    
    %Query the user for the label of this point
    set(hm, 'String', 'Press the left mouse button to label the red point as class 0 (Blue) or right button as 1 (Green)')
    h       = plot(x(ix), y(iy), 'rd', 'LineWidth', 2);
    [t1, t2, button] = ginput(1);
    delete(h)
    drawnow
    new_label = button == 3;
    
    %Add this point to the data with the supplied label
    new_features   = [x(ix); y(iy)]*ones(1,Weight) + randn(2,Weight)*0.001;
    train_features = [train_features, new_features];
    train_targets  = [train_targets, new_label*ones(1,Weight)];
    
    %Build a new decision region        
    D       = NearestNeighbor(train_features, train_targets, NN, region);
end
    
D = D > .5;
set(hm, 'String', '')

function D = NearestNeighbor(features, targets, NN, region)
%Find the nearest neighbor classifier according to the relative distances

M       = size(features,2);
N       = region(5);
x       = linspace(region(1),region(2),N);
y       = linspace(region(3),region(4),N);
D       = zeros(N);

if (M < NN),
   error('You specified more neighbors than there are points.')
end

if (NN < 3)
    error('Number of nearest neighbors must be at least 3 for this function to work')
end

y_dist	= (ones(N,1) * features(2,:) - y'*ones(1,M)).^2;

for i = 1:N,
    if (i/50 == floor(i/50)),
        disp(['Finished ' num2str(i) ' lines out of ' num2str(N) ' lines.'])
    end
    
    x_dist = ones(N,1)  * (features(1,:)-x(i)).^2;
    dist   = abs(x_dist + y_dist);   
    [sorted_dist, indices] = sort(dist');
    Tnearest = targets(indices(1:NN,:));
    Tdist    = sorted_dist(1:NN,:);
    D(:,i)   = (sum(Tnearest.*Tdist)./sum(Tdist))';  
    
end
