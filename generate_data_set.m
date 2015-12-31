function [features, targets] = generate_data_set(parameters, region)

% Generate a new data set given it's Gaussian parameters
% Inputs:
%	parameters: A structure containing:
%		m0, m1	- Means of the gaussians for both classes
%		s0, s1 	- Standard deviations of the gaussians for both classes
%		w0, w1 	- Weights of the gaussians for both classes
%		p0			- Probability of class 0
%	region	- Decision region vector: [-x x -y y number_of_points]
% 
% Outputs:
%	features - New features
%	targets  - New targets 

n0 = size(parameters.s0,1);
n1 = size(parameters.s1,1);

%Get the number of points from the user
N = str2num(char(inputdlg('Enter the number of points in data set:','Generate Data set')));

set(gcf,'pointer','watch');

%First, select the indices for class0 and the indices for class 1
N0 = round(parameters.p0*N);
N1 = N - N0;

indices  = randperm(N);
indices0 = indices(1:N0);
indices1 = indices(N0+1:N);

targets 				= zeros(1,N);
targets(indices1) = 1;
features 			= zeros(2,N);

%Now, make the points for class 0
%First, divide N0 to n0 pieces according to w0
parameters.w0 = round(N0 * parameters.w0);
if (sum(parameters.w0) ~= N0),
   parameters.w0(1) = parameters.w0(1) - (sum(parameters.w0) - N0);
end

cw0 = [0 ; cumsum(parameters.w0)];

for i = 1:n0,
   %Make w0(i) data points according to the distribution
   if (length(size(parameters.s0))>2),
      sigma = squeeze(parameters.s0(i,:,:));
   else
      sigma = parameters.s0;
   end
   A		= sigma;
   dist 	= A'*randn(2,parameters.w0(i));
   dist  = dist + parameters.m0(i,:)' * ones(1,parameters.w0(i));
   
   %Place them in one of the remaining places of indices0
	features(:,indices0(1+cw0(i):cw0(i+1))) = dist;
end


%Do the same for class 1
%First, divide N1 to n1 pieces according to w1
parameters.w1 = round(N1 * parameters.w1);
if (sum(parameters.w1) ~= N1),
   parameters.w1(1) = parameters.w1(1) - (sum(parameters.w1) - N1);
end
cw1 = [0 ; cumsum(parameters.w1)];

for i = 1:n1,
   %Make w1(i) data points according to the distribution
   if (length(size(parameters.s1))>2),
      sigma = squeeze(parameters.s1(i,:,:));
   else
      sigma = parameters.s1;
   end
   A		= sigma;
   dist 	= A'*randn(2,parameters.w1(i));
   dist  = dist + parameters.m1(i,:)' * ones(1,parameters.w1(i));
   
   %Place them in one of the remaining places of indices1
	features(:,indices1(1+cw1(i):cw1(i+1))) = dist;
end

N 		= region(5);
param = max(max(abs(features)));
plot_scatter(features, targets) 
axis([-param,param,-param,param])

set(gcf,'pointer','arrow');