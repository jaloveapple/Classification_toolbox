function a = predict_performance(algorithm, algorithm_params, features, targets, region)

% Predict the final performance of an algorithm from the learning curves
% Inputs:
%   algorithm           - The algorithm to test
%   algorithm_params    - Algorithm parameters
% 	features            - Train features
%	targets	            - Train targets
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	a		        	- Final performance prediction

[Ni, M] = size(features);

%Define the points in the data to check
Npoints = 10;
Nfold   = 5;
points  = linspace(0.05, 0.9, Npoints);

Etest   = zeros(Nfold, Npoints);
Etrain  = zeros(Nfold, Npoints);

%Train and test the classifiers
for i = 1:Npoints,
    for j = 1:Nfold,
        in      = randperm(M);
        train_in= in(1:floor(M*points(i)));
        test_in = in(floor(0.9*M)+1:end);
        D       = feval(algorithm, features(:, train_in), targets(train_in), algorithm_params, region);

        [classify, err]  = classification_error(D, features(:, train_in), targets(train_in), region);
        etrain(j,i)      = err;     
        [classify, err]  = classification_error(D, features(:, test_in), targets(test_in), region);
        etest(j,i)       = err;     
    end
    disp(['Finished ' num2str(i/Npoints*100) '% for this algorithm'])
        
end

%Find the parameters of the distribution
Etest   = mean(etest);
Etrain  = mean(etrain);

%alpha is the slope of the difference curve on a log-log scale, so:
p       = polyfit(log(floor(M*points)), log(Etest-Etrain),1);
alpha   = -p(2);

A       = [ones(Npoints,1) floor(M*points').^(-alpha) floor(M*points').^(-alpha)];
b       = (Etest+Etrain)';
X       = pinv(A)*b;

a       = real(X(1)/2);