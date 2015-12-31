function gamma = infomat(features, targets)

%Calculate the mutual information matrix (Use by the Koller algorithm)
%
%Inputs:
%	features - Input features
%  targets  - Input targets (0 or 1)
%
%Outputs:
%	gamma    - The information matrix

[Nf, L]	= size(features);
Kdist		= 2;				%How many features to group together
Nhist		= floor(L^(1/3));	%How many bins for the histogram

%Normalise features to [-1, 1]
features    = (2*features - (min(features')+max(features'))'*ones(1,L)) ./ ((max(features')-min(features'))'*ones(1,L));

gamma		= zeros(Nf);
in0		    = find(targets == 0);
in1		    = find(targets == 1);
P0			= length(in0);
P1			= length(in1);

features0   = features(:,in0);
features1   = features(:,in1);

disp('Started calculation of the cross-entropy matrix')

for i = 1:Nf,
   tic
   hist_i   = high_histogram(features(i,:),Nhist);
   g0d_i    = (high_histogram(features0(i,:),Nhist).*hist_i)*ones(1,Nhist);
   g1d_i    = (high_histogram(features1(i,:),Nhist).*hist_i)*ones(1,Nhist);   

   for j = i:Nf,
      hist_ij= high_histogram(features([i,j],:),Nhist);
      hist_j = high_histogram(features(j,:),Nhist);
      g0n 	 = high_histogram(features0([i,j],:),Nhist).*hist_ij;
      g0d_j  = (high_histogram(features0(j,:),Nhist).*hist_j)*ones(1,Nhist);
      g1n	 = high_histogram(features1([i,j],:),Nhist).*hist_ij;
      g1d_j	 = ((high_histogram(features1(j,:),Nhist).*hist_j)*ones(1,Nhist));
      
      %The addition of eps and the multipication by sign is for numeric reasons
      g0_j	 = g0n/P0.*log(eps + g0n./(g0d_j + eps)).*sign(g0d_j);
      g1_j   = g1n/P1.*log(eps + g1n./(g1d_j + eps)).*sign(g1d_j);
      gamma(i,j) = sum(sum(g0_j+g1_j));
      
      g0_i	 = g0n/P0.*log(eps + g0n./(g0d_i + eps)).*sign(g0d_i);
      g1_i   = g1n/P1.*log(eps + g1n./(g1d_i + eps)).*sign(g1d_i);
      gamma(j,i) = sum(sum(g0_i+g1_i));
   end
   
   t = toc;  
   disp(['Iteration ' num2str(i) ': Time taken: ' num2str(t) '[sec]'])
end
