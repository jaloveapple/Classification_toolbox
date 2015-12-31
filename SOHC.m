function [features, targets, label] = SOHC(train_features, train_targets, Nmu, region, plot_on)

%Reduce the number of data points using the stepwise optimal hierarchical clustering algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	Nmu 			- Number of output data points
%	region			- Decision region vector: [-x x -y y number_of_points]
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets
%	label				- The labels given for each of the original features

if (nargin < 5),
    plot_on = 0;
end

[D,L]	= size(train_features);
c_hat   = L;

if (Nmu == 1),
   mu		= mean(train_features')';
   label	= ones(1,c_hat);
else  
    m       = train_features;
    n       = ones(1,c_hat);
    
    while (c_hat ~= Nmu),
        c_hat = c_hat - 1;
        
        %Find two clusters j and i whose merger changes the criterion the least
        %The criterion is as in DHS, Chapter 10, eq. 83
        de  = zeros(c_hat+1);
        for i = 1:c_hat+1;
            de(i,:) = sqrt(sum((m(:,i)*ones(1,c_hat+1) - m).^2)).*sqrt(n(i)*n./(n+n(i)));
            de(i,i) = 1e30;
        end
        
        [i,j] = find(de == min(min(de)));
        i     = i(1);
        j     = j(1);
        
        %Merge clusters i and j (into cluster i in this realization)
        m(:,i)  = (n(i)*m(:,i) + n(j)*m(:,j))/(n(i) + n(j));
        n(i)    = n(i) + n(j);
        if (j < c_hat + 1),
            m       = m(:,[1:j-1,j+1:end]);
            n       = n(:,[1:j-1,j+1:end]);
        else
            m       = m(:,1:end-1);
            n       = n(1:end-1);
        end
        
        if (plot_on == 1),
            plot_process(m)
        else
            disp(['Reduced to ' num2str(c_hat) ' clusters so far'])
        end

    end
end

%Find labels for the examples
dist    = zeros(Nmu, L);
for i = 1:Nmu,
    dist(i,:) = sqrt(sum((m(:,i)*ones(1,L) - train_features).^2));
end
[temp, label]  = min(dist);

%Make the decision region
targets = zeros(1,Nmu);
if (Nmu > 1),
	for i = 1:Nmu,
   	if (length(train_targets(:,find(label == i))) > 0),
      	targets(i) = (sum(train_targets(:,find(label == i)))/length(train_targets(:,find(label == i))) > .5);
   	end
	end
else
   %There is only one center
   targets = (sum(train_targets)/length(train_targets) > .5);
end

features = m;