function D = parzen(train_features, train_targets, hn, region)

% Classify using the Parzen windows algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	hn      	- Normalizing factor for h
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

N		= region(5);								%Number of points on the grid
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);
Uc      = unique(train_targets);
V		= zeros(length(Uc), N, N);
x_i     = train_features;

for j = 1:length(Uc),
    indices = find(train_targets == Uc(j));
    P(j)    = length(indices)/size(x_i,2);
    n		= length(indices);
    
    for i = 1:n,
        temp        = (x - x_i(1,indices(i))).^2 + (y - train_features(2,indices(i))).^2;
        V(j,:,:)    = squeeze(V(j,:,:)) + phi(temp./hn);
        if (i/50 == floor(i/50)),
            disp(['Finished ' num2str(i) ' iterations out of ' num2str(n) ' iterations.'])
        end
        
    end
    
    V(j,:,:) = V(j,:,:) / sum(sum(squeeze(V(j,:,:))));
end


D = (squeeze(V(1,:,:))*P(1) < squeeze(V(2,:,:))*P(2));
%END Parzen

function p = phi(val)

%The window function for the Parzen window
p = (abs(val) <= 0.5);