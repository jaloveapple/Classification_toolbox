function D = CART(train_features, train_targets, params, region)

% Classify using classification and regression trees
% Inputs:
% 	features	- Train features
%	targets	    - Train targets
%	params		- [Impurity type, Percentage of incorrectly assigned samples at a node]
%                   Impurity can be: Entropy, Variance (or Gini), or Missclassification
%	region	    - Decision region vector: [-x x -y y number_of_points]
%
% Outputs
%	D			- Decision sufrace


[Ni, M]		   = size(train_features);

%Get parameters
[split_type, inc_node] = process_params(params);

%For the decision region
N           = region(5);
mx          = ones(N,1) * linspace (region(1),region(2),N);
my          = linspace (region(3),region(4),N)' * ones(1,N);
flatxy      = [mx(:), my(:)]';

%Preprocessing
[f, t, UW, m]   = PCA(train_features, train_targets, Ni, region);
train_features  = UW * (train_features - m*ones(1,M));;
flatxy          = UW * (flatxy - m*ones(1,N^2));;

%Build the tree recursively
disp('Building tree')
tree        = make_tree(train_features, train_targets, M, split_type, inc_node, region);

%Make the decision region according to the tree
disp('Building decision surface using the tree')
targets		= use_tree(flatxy, 1:N^2, tree);

D				= reshape(targets,N,N);
%END

function targets = use_tree(features, indices, tree)
%Classify recursively using a tree

if isnumeric(tree.Raction)
   %Reached an end node
   targets = zeros(1,size(features,2));
   targets(indices) = tree.Raction(1);
else
   %Reached a branching, so:
   %Find who goes where
   in_right    = indices(find(eval(tree.Raction)));
   in_left     = indices(find(eval(tree.Laction)));
   
   Ltargets	 = use_tree(features, in_left, tree.left);
   Rtargets	 = use_tree(features, in_right, tree.right);
   
   targets		 = Ltargets + Rtargets;
end
%END use_tree 

function tree = make_tree(features, targets, Dlength, split_type, inc_node, region)
%Build a tree recursively

if (length(unique(targets)) == 1),
   %There is only one type of targets, and this generates a warning, so deal with it separately
   tree.right      = [];
   tree.left       = [];
   tree.Raction    = targets(1);
   tree.Laction    = targets(1);
   break
end

[Ni, M] = size(features);
Nt      = unique(targets);
N       = hist(targets, Nt);

if ((sum(N < Dlength*inc_node) == length(Nt) - 1) | (M == 1)), 
   %No further splitting is neccessary
   tree.right      = [];
   tree.left       = [];
   if (length(Nt) ~= 1),
      MLlabel			  = find(N == max(N));
   else
      MLlabel 		  = 1;
   end
   tree.Raction    = Nt(MLlabel);
   tree.Laction    = Nt(MLlabel);
   
else
   %Split the node according to the splitting criterion  
   deltaI		= zeros(1,Ni);
   split_point = zeros(1,Ni);
   op				= optimset('Display', 'off');   
   for i = 1:Ni,
      split_point(i) = fminbnd('CARTfunctions', region(i*2-1), region(i*2), op, features, targets, i, split_type);
      I(i)				= feval('CARTfunctions', split_point(i), features, targets, i, split_type);
   end
   
   [m, dim]				= min(I);
   loc					= split_point(dim);
    
   %So, the split is to be on dimention 'dim' at location 'loc'
   indices		 = 1:M;
   tree.Raction= ['features(' num2str(dim) ',indices) >  ' num2str(loc)];
   tree.Laction= ['features(' num2str(dim) ',indices) <= ' num2str(loc)];
   in_right    = find(eval(tree.Raction));
   in_left     = find(eval(tree.Laction));
   
   if isempty(in_right) | isempty(in_left)
      %No possible split found
	   tree.right      = [];
   	tree.left       = [];
	   if (length(Nt) ~= 1),
   	   MLlabel  	 = find(N == max(N));
	   else
   	   MLlabel 		 = 1;
	   end
   	tree.Raction    = Nt(MLlabel);
   	tree.Laction    = Nt(MLlabel);
   else
	   %...It's possible to build new nodes
   	tree.right = make_tree(features(:,in_right), targets(in_right), Dlength, split_type, inc_node, region);
   	tree.left  = make_tree(features(:,in_left), targets(in_left), Dlength, split_type, inc_node, region);      
   end
   
end

