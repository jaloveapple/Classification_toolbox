function D = Marginalization(train_features, train_targets, missing, region)

% Classify data with missing features using the marginal distribution
% This file is strongly made for only two features
%
% Inputs:
% 	features			- Train features
%	targets			    - Train targets
%	missing 			- The number of the missing feature
%	region			    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

warning off;

[d, L] = size(train_features);
N		 = region(5);

if (missing > d),
    error(['The number of the missing feature must be between 1 and ' num2str(d)])
end

%Calculate the marginal distribution using histograms
Nbins   = max(3,floor(size(train_features,2).^(1/3)));
p       = high_histogram(train_features,Nbins,region(1:end-1));
classes = unique(train_targets);
for i = 1:length(classes),
    indices	= find(train_targets == classes(i));
    g_i     = high_histogram(train_features(:,indices),Nbins,region(1:end-1));
    if (missing == 1),
        P(i,:)  = sum(g_i.*p,missing)./sum(p,missing);
    else
        P(i,:)  = (sum(g_i.*p,missing)./sum(p,missing))';
    end
    bad = find(~isfinite(P(i,:)));
    P(i,bad) = 0;
end

%Build a decision region based on the marginal
decision    = P(2,:) > P(1,:);
if (missing == 1),
    y           = linspace(region(3),region(4),region(5));
    yy          = linspace(region(3),region(4),size(p,1));
    Decision    = interp1(yy, decision, y)>.5;
    D           = ones(region(5),1)*Decision;
else
    y           = linspace(region(1),region(2),region(5));
    yy          = linspace(region(1),region(2),size(p,1));
    Decision    = interp1(yy, decision, y)>.5;
    D           = Decision'*ones(1,region(5));
end            
D = D';
