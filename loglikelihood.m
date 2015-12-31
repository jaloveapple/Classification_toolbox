function ll = loglikelihood(theta, features, h, center_point, cp_target) 

% Used by the polynomial fitting algorithm

[c,r] = size(features);

features = center_point * ones(1,r) - features;

K = exp(-(features(1,:).^2+features(2,:).^2)/(2*h))/(2*pi*h);
f = 1./(1 + exp(-theta(1:2)'*features - theta(3)));
if isempty(find(f==0))
   L  = log(f);
   ll = -sum(K.*(cp_target.*L+~cp_target.*(1-L)));
else
   ll = nan;
end
