function [f, Df] = LocBoostFunctions(params, type, features, targets, h, params2)

%Return a value for the LocBoost algorithm functions

[c,r] = size(params);
r2		= r/2;
Nf		= size(features,2);

switch type
case 'class_kernel'
	%f = (1./(1 + exp((params* features).*targets)));
	f = (1./(1 + exp(-params* features))).^((1+targets)/2) .* ...
        ((1-1./(1 + exp(-params* features)))).^((1-targets)/2);
case 'Q1'
   f = -sum(h.*((1+targets)/2.*log(eps+(1./(1 + exp((-params* features))))) + ...
      			 (1-targets)/2.*log(eps+1-(1./(1 + exp((-params* features)))))));
case 'gamma_kernel'
	%f = (1./(1 + exp(params(1:r2) * features)));
    f = exp(-0.5*sum((params(1:r2)'*ones(1,Nf)-features).^2.*(params(r2+[1:r2])'*ones(1,Nf))));
case 'Q2'
   %f = -sum(h.*(log(eps+(1./(1 + exp((params(1:r2) * features))))) + ...
   %   			 (1-h).*log(eps+1-(1./(1 + exp((params(1:r2)* features)))))));
   f = -sum(h.*log(eps+exp(-0.5*sum((params(1:r2)'*ones(1,Nf)-features).^2.*(params(r2+[1:r2])'*ones(1,Nf))))) + ...
      (1-h).*log(eps+1-exp(-0.5*sum((params(1:r2)'*ones(1,Nf)-features).^2.*(params(r2+[1:r2])'*ones(1,Nf))))));
case 'NewTestSet'
    %This section is used for labeling new data (especially of dimension > 2)
    %In this case, params is phi and params2 is theta
    phi                 = params;
    theta               = params2;
    [Dims, Nf]          = size(features);
    targets             = ones(1,Nf);
    features(Dims+1,:)  = ones(1,length(targets));
    
    Pdecision           = LocBoostFunctions(theta(1,:), 'class_kernel', features, targets);
    
    for t = 2:size(params,1),
        Dgamma      = LocBoostFunctions(phi(t,:), 'gamma_kernel', features(1:Dims,:));  
        Dclass	    = LocBoostFunctions(theta(t,:), 'class_kernel', features, targets);
        Pdecision   = (1-Dgamma).*Pdecision + Dgamma.*Dclass;
    end
    
    f = Pdecision;
    %No nearest neighbor for 0.45<P<0.55 yet!!!
    
otherwise
   error ('Function type not recognized');
end


