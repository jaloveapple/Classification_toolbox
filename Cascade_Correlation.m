function D = Cascade_Correlation(train_features, train_targets, params, region)

% Classify using a backpropagation network with the cascade-correlation algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params   - Convergence criterion, Convergence rate
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

[Theta, eta] = process_params(params);
Nh		= 0;
iter	= 0;
Max_iter = 1e5;
NiterDisp = 10;

[Ni, M] = size(train_features);

%For the decision region
xx	= linspace(region(1),region(2),region(5));
yy	= linspace(region(3),region(4),region(5));
D  = zeros(region(5));

train_targets = (train_targets>0)*2-1;
means		  = mean(train_features')';
train_features= train_features - means*ones(1,M);

%Initialize the net: In this implementation there is only one output unit, so there
%will be a weight vector from the hidden units to the output units, and a weight matrix
%from the input units to the hidden units.
%The matrices are defined with one more weight so that there will be a bias
w0		= max(abs(std(train_features')'));
Wd		= rand(1,  Ni+1).*w0*2-w0;	%Direct unit weights
Wd      = Wd/mean(std(Wd'))*(Ni+1)^(-0.5);

rate	= 10*Theta;
J       = 1e3;

while ((rate > Theta) & (iter < Max_iter)),

    %Using batch backpropagation
    deltaWd	= 0;   
    for m = 1:M,
        Xm = train_features(:,m);
        tk = train_targets(m);
        
        %Forward propagate the input:
        %First to the hidden units
        gd			= Wd*[Xm; 1];
        [zk, dfo]	= activation(gd);
        
        %Now, evaluate delta_k at the output: delta_k = (tk-zk)*f'(net)
        delta_k		= (tk - zk).*dfo;
        
        deltaWd     = deltaWd + eta*delta_k*[Xm;1]';
        
    end

    %w_ki <- w_ki + eta*delta_k*Xm
    Wd				= Wd + deltaWd;
    
    %Calculate total error
    OldJ = J;
    J    = 0;
    for i = 1:M,
        J = J + (train_targets(i) - activation(Wd*[train_features(:,i);1])).^2;
    end
    J     = J/M; 
    rate  = abs(J - OldJ)/OldJ*100;

    iter 			= iter + 1;
    if (iter/NiterDisp == floor(iter/NiterDisp)),
        disp(['Direct unit, iteration ' num2str(iter) ': Average error is ' num2str(gradJ)])
    end
    
end

Wh	= rand(0, Ni+1).*w0*2-w0; %Hidden weights
Wo  = Wd;

while (gradJ > Theta),
    %Add a hidden unit
    Nh					= Nh + 1;
    Wh(Nh,:)			= rand(1, Ni+1).*w0*2-w0; %Hidden weights
    Wh(Nh,:)            = Wh(Nh,:)/std(Wh(Nh,:))*(Ni+1)^(-0.5);
    Wo(:,Ni+Nh+1)	    = rand(1, 1).*w0*2-w0; %Output weights
    
    gradJ			= 1e3;
    oldJ            = M;
    iter			= 0;
    
    rate	= 10*Theta;
    gradJ = 1e3;

    while ((rate > Theta) & (iter < Max_iter)),
        %Train each new unit with batch backpropagation
        deltaWo = 0;
        deltaWh = 0;
        for m = 1:M,
            Xm = train_features(:,m);
            tk = train_targets(m);

            %Find the output to this example
            y		= zeros(1, Ni+Nh+1);
            y(1:Ni)	= Xm;
            y(Ni+1) = 1;
            for i = 1:Nh,
                g		= Wh(i,:)*[Xm;1];
                if (i > 1),
                    g	= g - sum(y(Ni+2:Ni+i));
                end 
                [y(Ni+i+1), dfh]	= activation(g);
            end
        
            %Calculate the output
            go				= Wo*y';
            [zk, dfo]	= activation(go);
        
            %Evaluate the needed update
            delta_k		= (tk - zk).*dfo;
        
            %...and delta_j: delta_j = f'(net)*w_j*delta_k
            delta_j		= dfh.*Wo(end).*delta_k;
        
            deltaWo     = deltaWo + eta*delta_k*y(end);
            deltaWh		= deltaWh + eta*delta_j'*[Xm;1]';
        end
        
        %w_kj <- w_kj + eta*delta_k*y_j
        Wo(end)	    = Wo(end) + deltaWo;
        
        %w_ji <- w_ji + eta*delta_j*[Xm;1]
        Wh(Nh,:)		= Wh(Nh,:)  + deltaWh;
        
        %Calculate total error
        OldGradJ = gradJ;
        gradJ    = 0;
        for i = 1:M,
    	    Xm	    = train_features(:,i);
            gradJ   = gradJ + (train_targets(i) - cas_cor_activation(Xm, Wh, Wo, Ni, Nh)).^2;
        end
        gradJ = gradJ/M; 
        rate  = abs(gradJ - OldGradJ)/OldGradJ*100;
        
        iter 			= iter + 1;
        if (iter/NiterDisp == floor(iter/NiterDisp)),
            disp(['Hidden unit ' num2str(Nh) ', Iteration ' num2str(iter) ': Total error is ' num2str(gradJ)])
        end
    end
    
end

disp('Computing the decision region. This may take some time...')
%Find the decision region
for i = 1:region(5),
    for j = 1:region(5),
        Xm = [xx(i); yy(j)] - means;
        D(i,j) = cas_cor_activation(Xm, Wh, Wo, Ni, Nh);
    end
    if (floor(i/10) == i/10),
        disp(['Finished ' num2str(i) ' lines of ' num2str(region(5)) ' lines.'])
    end
end
D = D'>0;



function f = cas_cor_activation(Xm, Wh, Wo, Ni, Nh)

%Calculate the activation of a cascade-correlation network
y		= zeros(1, Ni+Nh+1);
y(1:Ni)	= Xm;
y(Ni+1) = 1;
for i = 1:Nh,
    g		= Wh(i,:)*[Xm;1];
    if (i > 1),
        g	= g - sum(y(Ni+2:Ni+i));
    end
    [y(Ni+i+1), dfh]	= activation(g);
end

%Calculate the output
go	= Wo*y';
f	= activation(go);


function [f, df] = activation(x)

a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
df	= a*b*sech(b*x).^2;