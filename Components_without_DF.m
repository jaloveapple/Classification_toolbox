function D = Components_without_DF(train_features, train_targets, Classifiers, region)

% Classify points using component classifiers with discriminant functions
% Inputs:
% 	train_features	- Train features
%	train_targets	- Train targets
%	Classifiers		- Classification algorithms. The format is:
%							[('<Algorithm 1>', <Algorithm 1 parameters>), ...]
%	region			- Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D					- Decision sufrace
%

%Read the algorithms
open_bracket	= findstr(Classifiers,'(');
close_bracket	= findstr(Classifiers,')');
if length(open_bracket) ~= length(close_bracket),
    error('Input vector contains an error!')
end
Nalgorithms		= length(open_bracket);
for i = 1:Nalgorithms,
    line	= Classifiers(open_bracket(i)+1:close_bracket(i)-1);
    comma = findstr(line,',');
    if isempty(comma),
        algorithms(i).name      = line(2:end-1);
        algorithms(i).params    = [];
    else
        algorithms(i).name      = line(2:comma-2);
        algorithms(i).params	= str2num(line(comma+1:end));
    end
end

%Train the weak classifiers
disp('Training weak classifiers')
[Ndim, M]		= size(train_features);
N				= region(5);
enum_features 	= round((train_features - region([1,3])'*ones(1,M))./((region([2,4])-region([1,3]))'*ones(1,M))*N);
p				= zeros(Nalgorithms, M);
Dweak			= zeros(Nalgorithms, N, N);
for i = 1:Nalgorithms,
    Dweak(i,:,:) = feval(algorithms(i).name, train_features, train_targets, algorithms(i).params, region);
end

for i = 1:M,
    p(:,i)	= squeeze(Dweak(:,enum_features(2,i), enum_features(1,i)));
end
p   = exp(p)./(1+exp(1));   %Use the softmax transformation of the data. We only have {0,1} classes, so the transformation is simple

%Init gating components
Ndim			= Ndim + 1;
x				= [train_features; ones(1,M)];
y				= train_targets;
alpha			= randn(Ndim, Nalgorithms);
alpha			= sqrtm(cov(x',1)+randn(Ndim))*alpha + mean(x')'*ones(1,Nalgorithms);
w   			= exp(alpha'*x)./(ones(Nalgorithms,1)*sum(exp(alpha'*x)));

%Learn the gating parameters
disp('Finding gating parameters')
old_err		= 10;
err			= 1;

while ((err > 1/M) & (err < old_err)),
    old_err = err;
    
    %Perform gradient descent on the gating parameters
    h               = w.*p./(ones(Nalgorithms,1)*sum(w.*p));
    dalpha          = (x*(h - w)');
    alpha			= alpha + dalpha;
    
    w				= exp(alpha'*x)./(ones(Nalgorithms,1)*sum(exp(alpha'*x)));
    Y				= sum(w.*p);
    err			= sum(y ~= (Y>.5))/M;
    
    disp(['Error is ' num2str(err)]) 
end

%Build decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:), ones(N^2,1)]';

u			= exp(alpha'*flatxy)./(ones(Nalgorithms,1)*sum(exp(alpha'*flatxy)));
U           = reshape(u,[Ndim, N, N]);
D			= reshape(squeeze(sum(U.*(exp(Dweak)/(1+exp(1)))))>.5,N,N);