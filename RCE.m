function D = RCE(train_features, train_targets, lambda_m, region)

% Classify using the reduced coulomb energy algorithm
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	lambda_m	- Maximum radius 
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

epsilon 	= 1e-4;
[Dim,Nf]    = size(train_features);
N			= region(5);
x           = linspace (region(1),region(2),N);
y           = linspace (region(3),region(4),N);
X           = train_features;

%Train the classifier
W   		= X;            %w_ij <- x_i
lambda      = zeros(1,Nf);

for i = 1:Nf,
    %x_hat <- arg min D(x, x_tag)
    dist 			 = sqrt(sum((X - X(:,i) * ones(1,Nf)).^2));
    [m, indices]      = sort(dist);
    x_hat			 = find(train_targets(indices) ~= train_targets(i));
    
    %lambda_j <- min(D(x_hat, x_tag)-epsilon, lambda_m)
    lambda(i)	     = min(dist(x_hat(1))-epsilon,lambda_m);
end

%Build the decision surface using the classifier
D = zeros(N);
for i = 1:N,
    for j = 1:N,
        %if D(x, x_hat_j)<lambda_j then D_t <- D_t U x_tag_j
        dist		 = sqrt(sum((X - [x(i) y(j)]' * ones(1,Nf)).^2));
        indices 		 = find(dist < lambda);
        %The decision is a little different from DH&S, since there an ambiguous result can
        %Occure. Here we do not allow this.
        if isempty(indices),
            D(j,i) = rand(1) > .5;
        else
            D(j,i) = sum(train_targets(indices))/length(indices) > .5;
        end
    end
    
    if (i/50 == floor(i/50)),
        disp(['Finished ' num2str(i) ' lines out of ' num2str(N) ' lines.']);
    end
end

