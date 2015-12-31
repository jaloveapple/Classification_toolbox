function D = Local_Polynomial(features, targets, Nlp, region)

% Classify using the local polynomial fitting
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Nlp		- Number of test points
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace

N		= region(5);
x		= linspace (region(1),region(2),N);
y		= linspace (region(3),region(4),N);
Dn		= zeros(N);
D		= zeros(N);
L		= length(targets);


%Choose h
Ntest  = Nlp;
Ntrain = L - Ntest;
[train_indices, test_indices] = make_a_draw(Ntest, L);

h = 0;
for i = 1:Ntest,
   dist  = sum((features(:,train_indices) - features(:,test_indices(i))*ones(1,Ntrain)).^2);
   dist  = sort(dist);
	new_h = dist(round(Ntrain/10))/2;
   if (new_h > h),
      h  = new_h;
   end
end

%Classify all the points in the decision region to one of the Ntest points
y_dist	= (ones(N,1) * features(2,test_indices) - y'*ones(1,Ntest)).^2;
for i = 1:N,
	if (i/50 == floor(i/50)),
      disp(['Finished ' num2str(i) ' lines out of ' num2str(N) ' lines.'])
   end

   x_dist = ones(N,1)  * (features(1,test_indices)-x(i)).^2;
   dist   = abs(x_dist + y_dist);   
   [sorted_dist, indices] = min(dist');
   Dn(:,i) = indices(1,:)';
end

%Now, built the plug-in classifier for each test point, and classify all the 
%points near it according to this classifier
mx		 = ones(N,1) * linspace (region(1),region(2),N);
my	    = linspace (region(3),region(4),N)' * ones(1,N);
flatDn = Dn(:);
flatD  = zeros(N^2,1);
flatxy = [mx(:)'; my(:)'];

for i = 1:Ntest,
   point   		 = features(:,test_indices(i));
   target_point = targets(:,test_indices(i));
   theta   		 = fminunc('loglikelihood',zeros(3,1),optimset('Display','off'),features(:,train_indices),h,point,target_point);
   indices = find(flatDn == i);
   X		  = features(:,test_indices(i))*ones(1,length(indices))- flatxy(:,indices);
   f_theta = 1./(1+exp(-theta(1:2)'*X-theta(3)));
   flatD(indices) = (f_theta > .5);
end

D = reshape(flatD,N,N);

