function D = PNN(train_features, train_targets, sigma, region)

% Classify using a probabilistic neural network
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	sigma   - Gaussian width
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[Dim, Nf]       = size(train_features);
Dim             = Dim + 1;
train_features(Dim,:) = ones(1,Nf);

%Build the classifier
x               = train_features;
W               = x ./ (ones(Dim,1)*sqrt(sum(x.^2)));  %x_jk <- x_jk / sqrt(sum(x_ji^2)), w_jk <- x_jk

%if x in w_i then a_ji <- 1
a(:,1)          = train_targets';                        
a(:,2)          = ~train_targets';                        

%Test it and build the decision region
%For decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:), ones(N^2,1)]';
flatxy      = flatxy ./ (ones(Dim,1)*sqrt(sum(flatxy.^2)));

%net_k <- W'_t*x
net         = W' * flatxy;

%if a_ki=1 then g_i <- g_i + exp((net-1)/sigma^2)
u_targets   = unique(train_targets);
arguments   = zeros(length(u_targets),size(flatxy,2));
for i = 1:length(u_targets),
    mask    = a(:,i) * ones(1,size(flatxy,2));
    arguments(i,:) = sum(exp((net-1)/sigma^2) .* mask);
end

%class <- argmax g(x)
[m, indices] = max(arguments);
targets      = zeros(1,size(flatxy,2));
for i = 1:length(u_targets),
    targets(find(indices == i)) = u_targets(i);
end

D = reshape(targets,N,N);

    