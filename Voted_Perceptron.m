function D = voted_perceptron(train_features, train_targets, params, region);

% Classify using the Perceptron algorithm
% Inputs:
% 	features	- Train features
%	targets	- Train targets
%	Params:
%			1. NumberOfPerceptrons
%			2. Kernel method (Linear, Polynomial, Gaussian)
%			3. Method's parameters ( Linear - none, Polinomial - power, Gaussian - sigma )
%	region	- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace
%
% Coded by: Igor Makienko and Victor Yosef

[NumberOfPerceptrons, method, alg_param] = process_params(params);

[c, n]		   = size(train_features);
train_features = [train_features ; ones(1,n)];
train_one      = find(train_targets == 1);
train_zero     = find(train_targets == 0);

%Preprocessing
processed_features = train_features;
processed_features(:,train_zero) = -processed_features(:,train_zero);


%Initial weights for Linear case:
w_percept  = rand(c+1,NumberOfPerceptrons);
%Initial alphas for kernel method:
alpha = rand(n,NumberOfPerceptrons);

%Initial permutation matrix for kernel case;

        switch method
        case 'Polynomial'
            perm = polyn(processed_features', processed_features',alg_param);
        case 'Gaussian'
            perm = gaus(processed_features',processed_features',alg_param);
        end
%Train targets for kernels' case [-1 1] :
t = 2 * train_targets - 1;
%Step for kernel case :
etta  = 1;
%Initial success vector:
w_sucesses = ones(NumberOfPerceptrons,1);

correct_classified = 0;
iter			   = 0;
max_iter		   = 500;

while (iter < max_iter)
   iter 		= iter + 1;
   indice 	= 1 + floor(rand(1)*n);
   switch method
   case 'Linear',
      InnerProduct = w_percept' * processed_features(:,indice);
      NegInnerProduct = (InnerProduct<=0);
      PosInnerProduct = (InnerProduct>0);
      w_sucesses = ones(size(w_sucesses)) + w_sucesses.*PosInnerProduct;
      w_percept(:,find(NegInnerProduct)) = w_percept(:,find(NegInnerProduct))...
         + processed_features(:,indice) * ones(1,sum(NegInnerProduct));    
     
   case {'Polynomial','Gaussian'}
      InnerProduct = perm(indice,:) * ((alpha'.*(ones(size(alpha,2),1)*t)))' ;     
      NegInnerProduct = (InnerProduct<=0)';
      PosInnerProduct = (InnerProduct>0)';
      w_sucesses = ones(size(w_sucesses)) + w_sucesses.*PosInnerProduct;
      alpha(indice,find(NegInnerProduct)) = alpha(indice,find(NegInnerProduct))...
         + etta * ones(1,sum(NegInnerProduct));
     otherwise
     error('Method unknown');
	end
end

if (iter == max_iter),
    disp(['Maximum iteration (' num2str(max_iter) ') reached'])
end
%Find decision region
N = region(5);
x = ones(N,1) * linspace (region(1),region(2),N);
y = linspace (region(3),region(4),N)' * ones(1,N);

D = zeros(N);

switch method
case 'Linear',
   for i = 1:NumberOfPerceptrons
      D = D + w_sucesses(i) * (2 * ((w_percept(1,i).*x + w_percept(2,i).*y + w_percept(c+1,i))> 0) - 1);
   end
case 'Polynomial',
    temp = [x(:),y(:),ones(size(y(:)))];
    perm  = polyn(temp,processed_features',alg_param);
    for i = 1:NumberOfPerceptrons,
        temp = 2 * (sum(((ones(N^2,1)*(alpha(:,i)'.* t)) .* (perm))') > 0) - 1;
        D = D + w_sucesses(i) * reshape(temp,N,N); 
    end

case 'Gaussian',
    temp = [x(:),y(:),ones(size(y(:)))];
    perm  = gaus(temp,processed_features',alg_param);
    for i = 1:NumberOfPerceptrons,
        temp = 2 * (sum(((ones(N^2,1)*(alpha(:,i)'.* t)) .* (perm))') > 0) - 1;
        D = D + w_sucesses(i) * reshape(temp,N,N); 
    end

end   
      
D = (D>0);

disp(['Iterated ' num2str(iter) ' times.'])

function out =  polyn(x,y,p);
%Routine function for polynomial kernel
%Input: 
%x - (number of vectors)x(dim+1) matrix  
%y - (number of vectors)x(dim+1) matrix
%p  - order of polynom
out = (ones(size(x,1),size(y,1)) + x * y').^p;

function out = gaus(x,y,sigma);
%Routine function for gaussian kernel
%Input: 
%x - (number of vectors)x(dim+1) matrix  
%y - (number of vectors)x(dim+1) matrix
%sigma  - std of gaussian kernel
x = x';y =y';c = [];
for i = 1:size(x,1),
    c(:,:,i) = (ones(size(x,2),1) * y(i,:) - x(i,:)' * ones(1,size(y,2))).^2;
end
out = exp( - squeeze( sum(permute(c,[3,1,2]))) ./ (2 * sigma) ^2);
        


