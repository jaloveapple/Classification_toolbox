function D = decision_region(param_struct, region)

%Function for making decision regions for Gaussians.
%Inputs are the means, covariances and weights for the Gaussians.
%Output is the decision region matrix, based on the "region" vector

%If class probabilities are not specified, assume equal distribution
if (~isfield(param_struct,'p0'))
   param_struct.p0 = 0.5; 
end

N		= region(length(region));								%Number of points on the grid
x		= ones(N,1) * linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N)' * ones(1,N);
V0		= zeros(N,N);
V1		= zeros(N,N);

n0 = length(param_struct.w0);
n1 = length(param_struct.w1);
disp(['Detected ' num2str(n0) ' Gaussians for class 0 and ' num2str(n1) ' Gaussians for class 1'])

for i = 1:n0,
   if (length(size(param_struct.s0))>2),
      sigma = squeeze(param_struct.s0(i,:,:));
   else
      sigma = param_struct.s0;
   end
   sigma = sigma.^2;
   if (param_struct.w0(i) ~= 0),
       invsigma = inv(sigma);
       V0 = V0 + param_struct.w0(i) ./ (2 * pi * sqrt(abs(det(sigma)))) .* ...
          exp(-0.5*(invsigma(1,1).*(x-param_struct.m0(i,1)).^2 + ...
          2*invsigma(2,1).*(x-param_struct.m0(i,1)).*(y-param_struct.m0(i,2))+invsigma(2,2).*(y-param_struct.m0(i,2)).^2));
   end
end

for i = 1:n1,
   if (length(size(param_struct.s1))>2),
      sigma = squeeze(param_struct.s1(i,:,:));
   else
      sigma = param_struct.s1;
   end
   sigma = sigma.^2;
   if (param_struct.w1(i) ~= 0),
       invsigma = inv(sigma);
       V1 = V1 + param_struct.w1(i) ./ (2 * pi * sqrt(abs(det(sigma)))) .* ...
            exp(-0.5*(invsigma(1,1).*(x-param_struct.m1(i,1)).^2 + ...
              2*invsigma(2,1).*(x-param_struct.m1(i,1)).*(y-param_struct.m1(i,2))+invsigma(2,2).*(y-param_struct.m1(i,2)).^2));
   end
end
    
D = (V0*param_struct.p0 < V1*(1-param_struct.p0));
