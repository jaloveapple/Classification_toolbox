function [D, Wh, Wo] = Backpropagation_SM(train_features, train_targets, params, region)

% Classify using a backpropagation network with stochastic learning algorithm with momentum
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params   - Number of hidden units, Convergence criterion, Convergence rate
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   Wh          - Hidden unit weights
%   Wo          - Output unit weights

[Nh, Theta, alpha, eta] = process_params(params);
iter	= 0;

[Ni, M] = size(train_features);
No		  = 1;

%For the decision region
xx	= linspace(region(1),region(2),region(5));
yy	= linspace(region(3),region(4),region(5));
D  = zeros(region(5));

train_targets = (train_targets>0)*2-1;
means			  = mean(train_features')';
train_features= train_features - means*ones(1,M);

%Initialize the net: In this implementation there is only one output unit, so there
%will be a weight vector from the hidden units to the output units, and a weight matrix
%from the input units to the hidden units.
%The matrices are defined with one more weight so that there will be a bias
w0		= max(abs(std(train_features')'));
Wh		= rand(Nh, Ni+1).*w0*2-w0; %Hidden weights
Wo		= rand(1,  Nh+1).*w0*2-w0; %Output weights
Wo    = Wo/mean(std(Wo'))*(Nh+1)^(-0.5);
Wh    = Wh/mean(std(Wh'))*(Ni+1)^(-0.5);

Bh    = zeros(size(Wh));
Bo    = zeros(size(Wo));

rate  = 10*Theta;
J     = 1e3;

while (rate > Theta),
    %Randomally choose an example
    i	= randperm(M);
    m	= i(1);
    Xm = train_features(:,m);
    tk = train_targets(m);
    
    %Forward propagate the input:
    %First to the hidden units
    gh				= Wh*[Xm; 1];
    [y, dfh]		= activation(gh);
    %Now to the output unit
    go				= Wo*[y; 1];
    [zk, dfo]	= activation(go);
    
    %Now, evaluate delta_k at the output: delta_k = (tk-zk)*f'(net)
    delta_k		= (tk - zk).*dfo;
    
    %...and delta_j: delta_j = f'(net)*w_j*delta_k
    delta_j		= dfh'.*Wo(1:end-1).*delta_k;
    
    %B_kj <- eta*(1-alpha)*delta_k*y_j + alpha*B_kj
    Bo				= eta*(1-alpha)*delta_k*[y;1]'+alpha*Bo;
    
    %B_ij <- eta*(1-alpha)*eta*delta_j*[Xm;1] + alpha*B_ij
    Bh				= eta*(1-alpha)*delta_j'*[Xm;1]' + alpha*Bh;
    
    %w_kj <- w_kj + B_kj
    Wo				= Wo + Bo;
    
    %w_ji <- w_ji + B_ji
    Wh				= Wh + Bh;
    
    %Calculate total error
    OldJ = J;
    J    = 0;
    for i = 1:M,
        J = J + (train_targets(i) - activation(Wo*[activation(Wh*[train_features(:,i); 1]); 1])).^2;
    end
    J = J/M; 
    rate  = abs(J - OldJ)/OldJ*100;
    
    iter 			= iter + 1;
    if (iter/100 == floor(iter/100)),
        disp(['Iteration ' num2str(iter) ': Total error is ' num2str(J)])
    end
    
end

disp(['Backpropagation converged after ' num2str(iter) ' iterations.'])

%Find the decision region
for i = 1:region(5),
    for j = 1:region(5),
        Xm = [xx(i); yy(j)] - means;
        D(i,j) = activation(Wo*[activation(Wh*[Xm; 1]); 1]);
    end
end
D = D'>0;

function [f, df] = activation(x)

a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
df	= a*b*sech(b*x).^2;