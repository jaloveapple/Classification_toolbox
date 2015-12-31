function [D, Wh, Wo] = Backpropagation_Recurrent(train_features, train_targets, params, region)

% Classify using a backpropagation recurrent network with a batch learning algorithm
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

[Nh, Theta, eta] = process_params(params);
[Ni, M] 	= size(train_features);
eta		= eta/M;
iter		= 0;
NiterDisp= 1;
maxIter 	= 100;

No		  = 1; %Number of output units

%For the decision region
xx	= linspace(region(1),region(2),region(5));
yy	= linspace(region(3),region(4),region(5));
D  = zeros(region(5));

train_targets = (train_targets>0)*2-1;
means			  = mean(train_features')';
train_features= train_features - means*ones(1,M);
train_features= [train_features; ones(1,M)];
Ni				  = Ni + 1;

%Init the network weights
w0		= max(abs(std(train_features')'));
W		= rand(Nh+No, Ni+No+Nh).*w0*2-w0; %The first Nh units are hidden, and the last No are the output
W     = W/mean(std(W'))*(Nh+No)^(-0.5);

rate	= 10*Theta;
J       = 1e3;

while (rate > Theta),
    %Randomally choose an example
    i	= randperm(M);
    m	= i(1);
    Xm = train_features(:,m);
    tk = train_targets(m);

    dOut    = 1;   %Fraction of change in output at time t
    dOutOld = 2;
    y_k     = zeros(Nh+No, 1);
    P_k     = zeros(Nh+No, Nh+No, Ni+Nh+No);
    U       = zeros(Nh+No, Nh+No, Ni+Nh+No);
    C       = [zeros(1, Nh) ones(1, No)];
    deltaW  = zeros(size(W));
    iter1    = 0;
    
    %Now, introduce the input to the net, and continue to feed it until it stabilizes
    while ((abs(dOut-dOutOld)/dOutOld > 1e-3) & (iter1 < maxIter)),
        iter1 = iter1 + 1;
        
        %Compute network output
        z_k                 = [Xm; y_k];
        net_k               = W*z_k;
        [y_k_tplus1, dy_k]  = activation(net_k);
        
        %Compute the error
        e_k                 = (tk - C*y_k_tplus1)';
        
        dOutOld             = dOut;
        dOut                = sum(abs(e_k./tk));
        
        %Build U and Phi
        for i = 1:Nh+No,
            U(i,i,:) = z_k';
        end
        Phi = eye(No+Nh) .* (dy_k*ones(1,No+Nh));
        
        %Compute the weight update
        for i = 1:Nh+No,
            deltaW(i,:)     = deltaW(i,:) + eta * C * squeeze(P_k(i,:,:)) * e_k;
        end
                
        %Update P_k
        for i = 1:Nh+No,
            P_k(i,:,:)      = Phi * (W(:,Ni+1:end) * squeeze(P_k(i,:,:)) + squeeze(U(i,:,:)));
        end

        y_k                 = y_k_tplus1;

    end

    %Update the weights
    W                   = W + deltaW;

    %Measure the error
    OldJ = J;
    J    = 0;
    for i = 1:M,
        Xm = train_features(:,i);
        tk = train_targets(i);

        dOut    = 1;   %Fraction of change in output at time t
        dOutOld = 2;
        y_k     = zeros(Nh+No, 1);
        iter1    = 0;
        
       %Now, introduce the input to the net, and continue to feed it until it stabilizes
        while ((abs(dOut-dOutOld)/dOutOld > 1e-3) & (iter1 < maxIter)),
            iter1 = iter1 + 1;
            %Compute network output
            z_k     = [Xm; y_k];
            net_k   = W*z_k;
            y_k     = activation(net_k);
            e_k     = (tk - C*y_k)';
            dOutOld = dOut;
            dOut    = sum(abs((tk-e_k)./tk));
        end
        
        J   = J + (tk - e_k).^2;        
    end
    J = J/M; 
    rate  = abs(J - OldJ)/OldJ*100;
               
    iter 			= iter + 1;
    if (iter/NiterDisp == floor(iter/NiterDisp)),
        disp(['Iteration ' num2str(iter) ': Error is ' num2str(J)])
    end
    
end

disp(['Backpropagation converged after ' num2str(iter) ' iterations.'])

%Find the decision region
for i = 1:region(5),
    for j = 1:region(5),
        Xm = [xx(i) - means(1); yy(j) - means(2); 1];
        
        dOut    = 1;   %Fraction of change in output at time t
        dOutOld = 2;
        y_k     = zeros(Nh+No, 1);
        iter1    = 0;
        
        %Now, introduce the input to the net, and continue to feed it until it stabilizes
        while ((abs(dOut-dOutOld)/dOutOld > 1e-3) & (iter1 < maxIter)),
            iter1 = iter1 + 1;
            
            %Compute network output
            z_k     = [Xm; y_k];
            net_k   = W*z_k;
            y_k     = activation(net_k);
            e_k     = (tk - C*y_k);
            dOutOld = dOut;
            dOut    = sum(abs((tk-e_k)./tk));
        end
        D(i,j) = y_k(end);
    end
end
D = D'>0;

function [f, df] = activation(x)

a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
df	= a*b*sech(b*x).^2;