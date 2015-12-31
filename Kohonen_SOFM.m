function [features, targets, label] = Kohonen_SOFM(train_features, train_targets, params, region, plot_on)

%Reduce the number of data points using a Kohonen self-orgenizing feature map algorithm
%Inputs:
%	train_features	- Input features
%	train_targets	- Input targets
%	params			- [Number of output data points, Window width]
%	region			- Decision region vector: [-x x -y y number_of_points]
%  plot_on         - Plot stages of the algorithm
%
%Outputs
%	features			- New features
%	targets			- New targets
%	label				- The labels given for each of the original features

[Nmu, win_width] = process_params(params);
if (nargin < 5),
    plot_on = 0;
end

[D,L]	= size(train_features);
dist	= zeros(Nmu,L);
label = zeros(1,L);

%Initialize W
W			= sqrtm(cov(train_features',1))*randn(D,Nmu);
W			= W ./ (ones(D,1)*sqrt(sum(W.^2)));
dW			= 1;

%Learning rate
eta		= 0.5;
deta		= 0.995;
iter		= 0;

while (dW > 1e-15),
   %Choose a sample randomally
   i		= randperm(L);
   phi	= train_features(:,i(1));
   
   net_k = W'*phi;
   y_star= find(net_k == max(net_k));
   y_star= y_star(1); %Just in case two have the same weights!
   
   oldW	= W;
   W		= W + eta*phi*gamma(win_width*abs(net_k - y_star))';
   W		= W ./ (ones(D,1)*sqrt(sum(W.^2)));
   
   eta	= eta * deta;
   
   dW		= sum(sum(abs(oldW-W)));
   iter	= iter + 1;   
   
   if (plot_on == 1),
      %Assign each of the features to a center
      dist        = W'*train_features;
      [m, label]  = max(dist);
      centers     = zeros(D,Nmu);
      for i = 1:Nmu,
         in = find(label == i);
         if ~isempty(in)
            centers(:,i) = mean(train_features(:,find(label==i))')';
         else
            centers(:,i) = nan;
         end
      end
      plot_process(centers)
   end
   
   if (iter/100 == floor(iter/100)),
      disp(['Iteration number ' num2str(iter)])
   end
   
end

%Assign a weight to each feature
label = zeros(1,L);
for i = 1:L,
   net_k 	= W'*train_features(:,i);
   label(i) = find(net_k == max(net_k));
end

%Find the target for each weight and the new features
targets 	= zeros(1,Nmu);
features	= zeros(D, Nmu);
for i = 1:Nmu,
   in				= find(label == i);
   if ~isempty(in),
      targets(i)		= sum(train_targets(in)) / length(in) > .5;
      if length(in) == 1,
         features(:,i)	= train_features(:,in);
      else
         features(:,i)  = mean(train_features(:,in)')';
      end
   end
   
end


function G = gamma(dist)
%The activation function for the SOFM
G = exp(-dist);