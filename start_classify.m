function [D, test_err, train_err, train_features, train_targets, reduced_features, reduced_targets] = start_classify(features, targets, error_method, redraws, percent, Preprocessing_algorithm, PreprocessingParameters, Classification_algorithm, AlgorithmParameters, region, hm, SepratePreprocessing, plot_on)

% Main function for evaluating a single classifier
% Inputs:
%   features                    - The examples of the data
%   targets                     - The labels for the data
%   error_method                - Error estimation method (Cross-validation, Holdout or Resubstitution)
%   redraws                     - Number of redraws needed
%   percent                     - Percentage of training vectors
%   Preprocessing_algorithm     - A preprocessing algorithm
%   PreprocessingParameters     - ...and it's parameters
%   Classification_algorithm    - A classification algorithm
%   AlgorithmParameters         - ...and it's parameters
%	region	                    - Decision region vector: [-x x -y y number_of_points]
%   hm                          - Handle to the message box on the GUI (Can be [])
%   SepratePreprocessing        - Perform separate preprocessing for each class
%   plot_on                     - Plot during preprocessing
%
% Outputs:
%   D                           - The last decision region
%   test_err                    - The test errors
%   train_err                   - The train errors
%   train_features              - The train features
%   train_targets               - ...and targets
%   reduced_features            - Features after preprocessing
%   reduced_targets             - ...and targets

%Some variable definitions
Nclasses		= find_classes(targets); %Number of classes in targets
test_err 		= zeros(Nclasses+1,redraws);   
train_err 		= zeros(Nclasses+1,redraws);   
x               = linspace(region(1), region(2), region(5));
y               = linspace(region(3), region(4), region(5));

reduced_features= [];
reduced_targets = [];

if ~isempty(hm),
    hParent         = get(hm,'Parent'); %Get calling window tag    
end

for i = 1: redraws,  
    if ~isempty(hm),
        set(hm, 'String', ['Processing iteration ' num2str(i) ' of ' num2str(redraws) ' iterations...']);
    end
   
   %Make a draw according to the error method chosen
   L = length(targets);
   switch error_method
   case cellstr('Resubstitution')
      test_indices = 1:L;
      train_indices = 1:L;
  	case cellstr('Holdout')
	   [test_indices, train_indices] = make_a_draw(floor(percent/100*L), L);           
   case cellstr('Cross-Validation')
      chunk = floor(L/redraws);
      test_indices = 1 + (i-1)*chunk : i * chunk;
      train_indices  = [1:(i-1)*chunk, i * chunk + 1:L];
   end
   train_features = features(:, train_indices);    
   train_targets  = targets (:, train_indices);    
   test_features  = features(:, test_indices);     
   test_targets   = targets (:, test_indices);     
   
   %Preprocess and then find decision region
   switch Preprocessing_algorithm
   case cellstr('None')
       disp('Generating decision region')
       D = feval(Classification_algorithm, train_features, train_targets, AlgorithmParameters, region); 
       disp('Calculating the error')
       [train_err(:,i), test_err(:,i)] = calculate_error (D, train_features, train_targets, test_features, test_targets, region, Nclasses);
       
   case cellstr('PCA')
       disp('Performing preprocessing')
       [reduced_features, reduced_targets, uw, m] = feval(Preprocessing_algorithm, train_features, train_targets, PreprocessingParameters, region); 
       reduced_features = uw*(train_features - m*ones(1,size(train_features,2)));
       disp('Generating decision region')
       [region, x, y] = calculate_region(uw*(features-m*ones(1,size(features,2))), region);    
       D = feval(Classification_algorithm, reduced_features, reduced_targets, AlgorithmParameters, region); 
       disp('Calculating the error')
       [train_err(:,i), test_err(:,i)] = calculate_error (D, reduced_features, reduced_targets, uw*(test_features-m*ones(1,size(test_features,2))), test_targets, region, Nclasses);      
       
   case cellstr('FishersLinearDiscriminant')
       disp('Performing preprocessing')
       [reduced_features, reduced_targets, w] = feval(Preprocessing_algorithm, train_features, train_targets, [], region); 
       [region, x, y] = calculate_region(reduced_features, region);    
       disp('Generating decision region')
       D = feval(Classification_algorithm, reduced_features, reduced_targets, AlgorithmParameters, region); 
       disp('Calculating the error')
       [train_err(:,i), test_err(:,i)] = calculate_error (D, reduced_features, reduced_targets, [w'*test_features; zeros(1,length(test_targets))], test_targets, region, Nclasses);      

       %If possible, replot the data
       if ~isempty(hParent),
           hold off
           plot_scatter([w'*features; zeros(1,length(targets))], targets, hParent)
           hold on
       end
       
   otherwise
      disp('Performing preprocessing')
      if SepratePreprocessing,
         disp('Perform seperate preprocessing for each class.')
         in0 = find(train_targets == 0);
         in1 = find(train_targets == 1);
       	 [reduced_features0, reduced_targets0] = feval(Preprocessing_algorithm, train_features(:,in0), train_targets(in0), PreprocessingParameters, region, plot_on); 
         [reduced_features1, reduced_targets1] = feval(Preprocessing_algorithm, train_features(:,in1), train_targets(in1), PreprocessingParameters, region, plot_on); 
         reduced_features = [reduced_features0, reduced_features1];
         reduced_targets  = [reduced_targets0,  reduced_targets1];
	   else
         [reduced_features, reduced_targets] = feval(Preprocessing_algorithm, train_features, train_targets, PreprocessingParameters, region, plot_on); 
      end
      pause(1);
      plot_process([]);
      indices = find(sum(isfinite(reduced_features)) > 0);
      reduced_features = reduced_features(:,indices);
      reduced_targets  = reduced_targets(:,indices);
      if ((i == redraws) & (~isempty(hParent)))
         %Plot only during the last iteration
         plot_scatter(reduced_features, reduced_targets, hParent, 1)
	      axis(region(1:4))
      end
      %Show Voronoi diagram
      if ~isempty(findobj('Tag','Voronoi diagram')),
         %Voronoi diagram figure exists
         figure(findobj('Tag','Voronoi diagram'))
         clf;
      else
         figure;
         set(gcf,'Tag','Voronoi diagram');
      end
      hold on
      contour(x,y,voronoi_regions(reduced_features, region),length(reduced_targets))
      plot_scatter(reduced_features, reduced_targets)
      hold off
      axis(region(1:4));
      grid on;
      title('Voronoi regions')
      if ~isempty(hParent),
          figure(hParent)
      end
      if ((sum(reduced_targets) <= 1) & (sum(~reduced_targets) <= 1) & (~strcmp(Classification_algorithm,'None')))
            error('Too few reduced points (This program needs at least two points of each class). Please restart.')
      else
        if strcmp(Classification_algorithm,'None'),
            %No classification was asked for
            D = zeros(region(5));
            set(gcf,'pointer','arrow');     
        end
      end
	  disp('Generating decision region')
      D = feval(Classification_algorithm, reduced_features, reduced_targets, AlgorithmParameters, region); 
      disp('Calculating the error')
      [train_err(:,i), test_err(:,i)] = calculate_error (D, train_features, train_targets, test_features, test_targets, region, Nclasses);
      
   end      
  
end      
  
