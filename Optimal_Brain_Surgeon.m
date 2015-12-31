function [D, Wh, Wo] = Optimal_Brain_Surgeon(train_features, train_targets, params, region)

% Classify using a backpropagation network with a batch learning algorithm and remove excess units
% using the optimal brain surgeon algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params   - Initial number of hidden units, Convergence criterion
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%   Wh          - Hidden unit weights
%   Wo          - Output unit weights

[Nh, Theta] = process_params(params);
[Ni, M]		= size(train_features);

%Train a reasonably large network to minimum error
disp(['Training a neural network with ' num2str(Nh) ' hidden units']);
[D, Wh, Wo] = Backpropagation_Batch(train_features, train_targets, [Nh, Theta, 0.1], region);

train_targets = (train_targets>0)*2-1;
means		  = mean(train_features')';
train_features= train_features - means*ones(1,M);

J             = 0;
gradJ         = 0;

%Cascade all the weights
W       = [Wo(:); Wh(:)];

%Define initial H's
alpha   = 1e-8;
invH    = (alpha^-1)*eye(length(W));         
Lq      = [1 0];
pruned  = 1:length(W);

disp('Pruning excess units');
while ((gradJ < Theta) & (length(unique(Lq)) > 1)),

    %Compute inv(H) by equation 70, DHS chapter 6
    for i = 1:M,
        xm = train_features(:,i);
        tk = train_targets(i);
        
        %Forward propagate the input:
        %First to the hidden units
        gh				= Wh*[xm; 1];
        [y, dfh]		= activation(gh);
        
        %Now to the output unit
        go				= Wo*[y; 1];
        [zk, dfo]	    = activation(go);
  
        Ni              = size(xm,1)+1;
        Xv              = dfo*[y; 1];
        Xu              = dfo*(dfh*ones(1,Ni)).*Wh.*(y*ones(1,Ni));

        Xm              = [Xv; Xu(:)];

        invH            = invH + (invH*Xm*Xm'*invH)/(M + Xm'*invH*Xm);
    end
    
    if ~isfinite(sum(sum(invH))),
        break
    end
    
    %q* <- argmin(w_q^2/(2*inv(H)_qq))
    Lq          = W.^2./(2*diag(invH));
    [m, q_star] = min(Lq(pruned)); %We don't want to prune the same weight twice
    q_star      = pruned(q_star);
    e_q_star    = zeros(length(W), 1);
    e_q_star(q_star) = 1;
    
    pruned = pruned(find(pruned ~= q_star));
    
    %w  <- w - w*_q/inv(H)_q*q**inv(H)*e_q*
    W           = W - W(q_star)/invH(q_star, q_star)*invH*e_q_star;
    
    Wo          = W(1:size(Wo,2))';
    Wh          = reshape(W(size(Wo,2)+1:end),size(Wh));

    %Calculate total error
    OldJ = J;
    J    = 0;
    for i = 1:M,
        J = J + (train_targets(i) - activation(Wo*[activation(Wh*[train_features(:,i); 1]); 1])).^2;
    end
    J = J/M; 
    if (OldJ == 0),
        gradJ = 0;
    else
        gradJ = J - OldJ;
    end

    disp(['Removed weight number ' num2str(q_star) '. Gradient jump was ' num2str(gradJ)]);

end

%Find the decision region
xx	= linspace(region(1),region(2),region(5));
yy	= linspace(region(3),region(4),region(5));
D  = zeros(region(5));

for i = 1:region(5),
    for j = 1:region(5),
        Xm = [xx(i); yy(j)] - means;
        D(i,j) = activation(Wo*[activation(Wh*[Xm; 1]); 1]);
    end
end
D = D'>0;


function [f, df] = activation(x)
%The activation function for a neural network
a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
df	= a*b*sech(b*x).^2;