function [D, Wh, Wo] = Backpropagation_Quickprop(train_features, train_targets, params, region)

% Classify using a backpropagation network with a batch learning algorithm and quickprop
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params   - Number of hidden units, Convergence criterion, Convergence rate, mu
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   Wh          - Hidden unit weights
%   Wo          - Output unit weights

% The basic idea in quickprop is that the update rule is changed so that:
% delta_w <- delta_J(m)/(delta_J(m-1)-delta_J(m))*delta_w(m)
% and this is done for each weight separately

[Nh, Theta, eta, mu] = process_params(params);
iter	= 0;
IterDisp= 10;

[Ni, M] = size(train_features);
No      = 1;

%For the decision region
xx	= linspace(region(1),region(2),region(5));
yy	= linspace(region(3),region(4),region(5));
D  = zeros(region(5));

train_targets = (train_targets>0)*2-1;
means	      = mean(train_features')';
train_features= train_features - means*ones(1,M);

%Initialize the net: In this implementation there is only one output unit, so there
%will be a weight vector from the hidden units to the output units, and a weight matrix
%from the input units to the hidden units.
%The matrices are defined with one more weight so that there will be a bias
w0			= max(abs(std(train_features')'));
Wh			= rand(Nh, Ni+1).*w0*2-w0; %Hidden weights
Wo			= rand(No,  Nh+1).*w0*2-w0; %Output weights
Wo          = Wo/mean(std(Wo'))*(Nh+1)^(-0.5);
Wh          = Wh/mean(std(Wh'))*(Ni+1)^(-0.5);
OldDeltaWo  = zeros(size(Wo));
OldDeltaWh  = zeros(size(Wh));
deltaJo     = zeros(size(Wo));
deltaJh     = zeros(size(Wh));
OldDeltaJo  = zeros(size(Wo));
OldDeltaJh  = zeros(size(Wh));

J          	= 1e3;
rate        = Theta*10;

while (rate > Theta),
    OldDeltaJo  = deltaJo;
    OldDeltaJh  = deltaJh;
    deltaJo     = zeros(size(Wo));
    deltaJh     = zeros(size(Wh));
    
    for m = 1:M,
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
        
        %delta_w_kj <- w_kj + eta*delta_k*y_j
        deltaJo		= deltaJo + delta_k*[y;1]';
        
        %delta_w_ji <- w_ji + eta*delta_j*[Xm;1]
        deltaJh		= deltaJh + delta_j'*[Xm;1]';
        
    end
    
    %delta_w <- delta_J(m)/(delta_J(m-1)-delta_J(m))*delta_w(m)
    %Well, it's not that simple. For details see "Back Propagation Family Album" by Jondarr Gibb. 
    %Dept. of Computing, Macquarie University, Technical report C/TR95-05, 1996.
    deltaWo     = zeros(size(Wo));
    deltaWh     = zeros(size(Wh));
    for i = 1:size(Wo,1),
        for j = 1:size(Wo,2),
            if (OldDeltaWo(i,j) > 0),
                if (deltaJo(i,j) > 0),
                    deltaWo(i,j) = eta * deltaJo(i,j);
                end
                if (deltaJo(i,j) > mu/(mu+1)*OldDeltaJo(i,j)),
                    deltaWo(i,j) = deltaWo(i,j) + mu*OldDeltaWo(i,j);
                else
                    deltaWo(i,j) = deltaWo(i,j) + deltaJo(i,j) * OldDeltaWo(i,j) / (OldDeltaJo(i,j) - deltaJo(i,j));
                end
            else
                if (OldDeltaWo(i,j) < 0),
                    if (deltaJo(i,j) < 0),
                        deltaWo(i,j) = eta * deltaJo(i,j);
                    end
                    if (deltaJo(i,j) < mu/(mu+1)*OldDeltaJo(i,j)),
                        deltaWo(i,j) = deltaWo(i,j) + mu*OldDeltaWo(i,j);
                    else
                        deltaWo(i,j) = deltaWo(i,j) + deltaJo(i,j) * OldDeltaWo(i,j) / (OldDeltaJo(i,j) - deltaJo(i,j));
                    end
                else
                    deltaWo(i,j) = eta * deltaJo(i,j);
                end
            end
        end
    end
    for i = 1:size(Wh,1),
        for j = 1:size(Wh,2),
            if (OldDeltaWh(i,j) > 0),
                if (deltaJh(i,j) > 0),
                    deltaWh(i,j) = eta * deltaJh(i,j);
                end
                if (deltaJh(i,j) > mu/(mu+1)*OldDeltaJh(i,j)),
                    deltaWh(i,j) = deltaWh(i,j) + mu*OldDeltaWh(i,j);
                else
                    deltaWh(i,j) = deltaWh(i,j) + deltaJh(i,j) * OldDeltaWh(i,j) / (OldDeltaJh(i,j) - deltaJh(i,j));
                end
            else
                if (OldDeltaWh(i,j) < 0),
                    if (deltaJh(i,j) < 0),
                        deltaWh(i,j) = eta * deltaJh(i,j);
                    end
                    if (deltaJh(i,j) < mu/(mu+1)*OldDeltaJh(i,j)),
                        deltaWh(i,j) = deltaWh(i,j) + mu*OldDeltaWh(i,j);
                    else
                        deltaWh(i,j) = deltaWh(i,j) + deltaJh(i,j) * OldDeltaWh(i,j) / (OldDeltaJh(i,j) - deltaJh(i,j));
                    end
                else
                    deltaWh(i,j) = eta * deltaJh(i,j);
                end
            end
        end
    end
    
    Wo = Wo + deltaWo;
    Wh = Wh + deltaWh;
    
    OldDeltaWo = deltaWo;
    OldDeltaWh = deltaWh;
    
    %Calculate total error
    oldJ = J;
    J    = 0;
    for i = 1:M,
        J = J + ((train_targets(i) - activation(Wo*[activation(Wh*[train_features(:,i); 1]); 1])).^2);
    end
    J    = J/M;
    rate = abs(J - oldJ)/oldJ*100;

    iter 			= iter + 1;
    if (iter/IterDisp == floor(iter/IterDisp)),
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
if (sum(sum(D)) == 0),
    D = D - mean(mean(D));
else
    D = D'>0;
end

function [f, df] = activation(x)

a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
df	= a*b*sech(b*x).^2;